"""通用 PyTorch trace 框架。

负责：
1. 用全局 module hook 捕获每个 nn.Module 的 forward 调用（输入/输出 tensor shape）。
2. 通过 monkey-patch 在 `make_policy` 返回时抓住 policy 对象，事后可以从 `policy.named_modules()`
   把"匿名 id"映射回语义化的层级名（如 `model.vlm_with_expert.vision_tower.encoder.layers.0`）。
3. 用 sentinel 机制在第一次 `Optimizer.step()` 完成后抛 `TraceComplete`，wrapper 捕获后落 JSON。
4. 支持 functional ops 的 monkey-patch hook（v0 仅 `F.scaled_dot_product_attention`），用来抓
   attention QKV 这类不走 nn.Module 的关键算子。
5. 支持 collate_fn 的 wrap，记录"原始 batch"的结构。

不修改 lerobot / SmolVLA 源码。
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Sentinel: 让 trace_runner 知道"采够了, 该收尾"
# ---------------------------------------------------------------------------
class TraceComplete(Exception):
    """正常的中断信号 —— trace 采够了, 把训练循环立即退出。"""


# ---------------------------------------------------------------------------
# Shape 编码: 把 inputs/outputs 递归压成 JSON-safe 结构
# ---------------------------------------------------------------------------
def _encode_value(x: Any, _depth: int = 0) -> Any:
    """递归把 tensor/list/tuple/dict 转成 JSON-safe 描述, 只记结构不记值。"""
    if _depth > 6:
        return {"_type": "truncated"}
    if isinstance(x, torch.Tensor):
        return {
            "_type": "Tensor",
            "shape": list(x.shape),
            "dtype": str(x.dtype).replace("torch.", ""),
            "device": str(x.device),
        }
    if isinstance(x, (list, tuple)):
        return {
            "_type": type(x).__name__,
            "items": [_encode_value(v, _depth + 1) for v in x[:32]],
            "len": len(x),
        }
    if isinstance(x, dict):
        return {
            "_type": "dict",
            "items": {str(k): _encode_value(v, _depth + 1) for k, v in list(x.items())[:64]},
            "len": len(x),
        }
    if x is None:
        return {"_type": "NoneType"}
    if isinstance(x, (int, float, bool, str)):
        s = repr(x)
        return {"_type": type(x).__name__, "repr": s[:120]}
    # fallback
    return {"_type": type(x).__name__, "repr": repr(x)[:120]}


# ---------------------------------------------------------------------------
# 事件 schema
# ---------------------------------------------------------------------------
@dataclass
class ModuleEvent:
    event_idx: int
    depth: int
    module_id: int
    qualname: str            # type(module).__module__ + "." + type(module).__name__
    src_file: str | None
    src_line: int | None
    phase: str               # "forward_pre" | "forward_post" | "backward_pre" | "backward_post"
    inputs: Any = None       # forward_pre / backward_pre
    outputs: Any = None      # forward_post / backward_post
    duration_ms: float | None = None  # 只在 forward_post 上有, 表示这次 forward 用了多久
    extra: dict = field(default_factory=dict)


@dataclass
class FunctionalEvent:
    event_idx: int
    depth: int
    op_name: str             # e.g. "F.scaled_dot_product_attention"
    src_file: str | None
    src_line: int | None
    inputs: Any
    outputs: Any
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tracer 主类: 全局状态 + hook 注册
# ---------------------------------------------------------------------------
class Tracer:
    def __init__(
        self,
        output_dir: Path,
        max_optimizer_steps: int = 1,
        capture_backward: bool = False,  # SmolVLA forward 含 inplace, 默认关闭 backward
        capture_functional: bool = True,
        verbose: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_optimizer_steps = max_optimizer_steps
        self.capture_backward = capture_backward
        self.capture_functional = capture_functional
        self.verbose = verbose

        # event tracking
        self._lock = threading.Lock()
        self.module_events: list[ModuleEvent] = []
        self.functional_events: list[FunctionalEvent] = []
        self._event_counter = 0
        self._depth = 0
        self._forward_start_times: dict[int, float] = {}  # module id -> perf_counter

        # captured artifacts (filled by monkey-patches)
        self.captured_policy: nn.Module | None = None
        self.captured_dataset_meta: dict | None = None
        self.captured_first_batch: Any = None
        self.optimizer_step_count = 0

        # source-file resolution cache
        self._src_cache: dict[type, tuple[str | None, int | None]] = {}

        # hook handles (so we can clean up)
        self._handles: list[Any] = []
        self._patched_originals: list[tuple[object, str, Any]] = []

    # ------------------------------------------------------------------
    # event index helper (locked)
    # ------------------------------------------------------------------
    def _next_idx(self) -> int:
        idx = self._event_counter
        self._event_counter += 1
        return idx

    # ------------------------------------------------------------------
    # source location lookup (cached per class)
    # ------------------------------------------------------------------
    def _src_loc(self, module: nn.Module) -> tuple[str | None, int | None]:
        cls = type(module)
        if cls in self._src_cache:
            return self._src_cache[cls]
        try:
            src_file = inspect.getsourcefile(cls)
            _, src_line = inspect.getsourcelines(cls)
        except (OSError, TypeError):
            src_file, src_line = None, None
        self._src_cache[cls] = (src_file, src_line)
        return src_file, src_line

    # ------------------------------------------------------------------
    # forward / backward hooks
    # ------------------------------------------------------------------
    def _forward_pre_hook(self, module: nn.Module, inputs: tuple) -> None:
        with self._lock:
            idx = self._next_idx()
            depth = self._depth
            self._depth += 1
            self._forward_start_times[id(module)] = time.perf_counter()
            src_file, src_line = self._src_loc(module)
            self.module_events.append(ModuleEvent(
                event_idx=idx,
                depth=depth,
                module_id=id(module),
                qualname=f"{type(module).__module__}.{type(module).__name__}",
                src_file=src_file,
                src_line=src_line,
                phase="forward_pre",
                inputs=_encode_value(inputs),
            ))

    def _forward_post_hook(self, module: nn.Module, inputs: tuple, outputs: Any) -> None:
        with self._lock:
            self._depth -= 1
            t0 = self._forward_start_times.pop(id(module), None)
            duration_ms = (time.perf_counter() - t0) * 1000.0 if t0 is not None else None
            idx = self._next_idx()
            src_file, src_line = self._src_loc(module)
            self.module_events.append(ModuleEvent(
                event_idx=idx,
                depth=self._depth,
                module_id=id(module),
                qualname=f"{type(module).__module__}.{type(module).__name__}",
                src_file=src_file,
                src_line=src_line,
                phase="forward_post",
                outputs=_encode_value(outputs),
                duration_ms=duration_ms,
            ))
            # Tensor-level backward hook: 不走 nn.Module 的 backward hook 路径（那条会
            # 给 module output 包 BackwardHookFunctionBackward, 跟下游 inplace ops
            # 冲突），而是直接给 tensor 自身注册 grad 回调。
            if self.capture_backward:
                self._attach_tensor_backward_hooks(module, outputs, idx)

    def _attach_tensor_backward_hooks(self, module: nn.Module, outputs: Any, post_idx: int) -> None:
        """递归找 outputs 里所有 requires_grad 的 Tensor, 给它们挂 grad hook."""
        mid = id(module)
        qual = f"{type(module).__module__}.{type(module).__name__}"

        def make_cb(tensor_role: str):
            def cb(grad: torch.Tensor) -> None:
                try:
                    with self._lock:
                        bidx = self._next_idx()
                        self.module_events.append(ModuleEvent(
                            event_idx=bidx,
                            depth=-1,  # backward 时 depth 已无意义
                            module_id=mid,
                            qualname=qual,
                            src_file=None,
                            src_line=None,
                            phase="backward",
                            inputs=_encode_value(grad),
                            extra={"tensor_role": tensor_role, "forward_post_idx": post_idx},
                        ))
                except Exception:
                    pass
            return cb

        def walk(value, path: str = "out"):
            if isinstance(value, torch.Tensor):
                if value.requires_grad and value.grad_fn is not None:
                    try:
                        value.register_hook(make_cb(path))
                    except Exception:
                        pass
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value[:8]):
                    walk(v, f"{path}[{i}]")
            elif isinstance(value, dict):
                for k, v in list(value.items())[:8]:
                    walk(v, f"{path}[{k}]")

        walk(outputs)

    def _backward_pre_hook(self, module: nn.Module, grad_output: tuple) -> None:
        with self._lock:
            idx = self._next_idx()
            src_file, src_line = self._src_loc(module)
            self.module_events.append(ModuleEvent(
                event_idx=idx,
                depth=self._depth,
                module_id=id(module),
                qualname=f"{type(module).__module__}.{type(module).__name__}",
                src_file=src_file,
                src_line=src_line,
                phase="backward_pre",
                inputs=_encode_value(grad_output),
            ))

    def _backward_post_hook(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        with self._lock:
            idx = self._next_idx()
            src_file, src_line = self._src_loc(module)
            self.module_events.append(ModuleEvent(
                event_idx=idx,
                depth=self._depth,
                module_id=id(module),
                qualname=f"{type(module).__module__}.{type(module).__name__}",
                src_file=src_file,
                src_line=src_line,
                phase="backward_post",
                inputs=_encode_value(grad_output),
                outputs=_encode_value(grad_input),
            ))

    # ------------------------------------------------------------------
    # monkey-patch hooks: functional ops, optimizer, make_policy
    # ------------------------------------------------------------------
    def _patch_attribute(self, owner: object, name: str, new: Any) -> None:
        original = getattr(owner, name)
        self._patched_originals.append((owner, name, original))
        setattr(owner, name, new)

    def _install_functional_hooks(self) -> None:
        """v0: 只 wrap scaled_dot_product_attention —— SmolVLA / SmolVLM 用了大量。"""
        if not self.capture_functional:
            return
        original_sdpa = F.scaled_dot_product_attention
        tracer = self

        def wrapped_sdpa(*args, **kwargs):
            with tracer._lock:
                idx = tracer._next_idx()
                depth = tracer._depth
                # caller location
                try:
                    frame = inspect.currentframe().f_back
                    src_file = frame.f_code.co_filename
                    src_line = frame.f_lineno
                except Exception:
                    src_file, src_line = None, None
                inputs_enc = _encode_value({"args": args, "kwargs": kwargs})
            outputs = original_sdpa(*args, **kwargs)
            with tracer._lock:
                tracer.functional_events.append(FunctionalEvent(
                    event_idx=idx,
                    depth=depth,
                    op_name="F.scaled_dot_product_attention",
                    src_file=src_file,
                    src_line=src_line,
                    inputs=inputs_enc,
                    outputs=_encode_value(outputs),
                ))
            return outputs

        self._patch_attribute(F, "scaled_dot_product_attention", wrapped_sdpa)

    def _install_optimizer_sentinel(self) -> None:
        """走 lerobot.make_optimizer_and_scheduler 抓住 optimizer 实例,
        通过 `Optimizer.register_step_post_hook`（torch ≥ 1.13）注册回调。

        为什么不 patch `Optimizer.step`：AdamW 等子类有自己的 `step` 覆盖, base class patch 不会被调用。
        instance-level register_step_post_hook 是官方 API, 任何 optimizer 子类都生效。
        """
        try:
            from lerobot.optim import factory as opt_factory
        except ImportError:
            if self.verbose:
                print("[smolvla-trace] lerobot.optim.factory not importable; skipping optimizer hook")
            return

        original_make = opt_factory.make_optimizer_and_scheduler
        tracer = self

        def post_step_hook(optimizer, args, kwargs):
            tracer.optimizer_step_count += 1
            if tracer.verbose:
                print(f"[smolvla-trace] optimizer.step #{tracer.optimizer_step_count}", flush=True)
            if tracer.optimizer_step_count >= tracer.max_optimizer_steps:
                raise TraceComplete(
                    f"completed {tracer.optimizer_step_count} optimizer step(s)"
                )

        def wrapped_make_optimizer(*args, **kwargs):
            optimizer, scheduler = original_make(*args, **kwargs)
            try:
                optimizer.register_step_post_hook(post_step_hook)
                if tracer.verbose:
                    print(f"[smolvla-trace] registered step_post_hook on {type(optimizer).__name__}",
                          flush=True)
            except Exception as e:
                if tracer.verbose:
                    print(f"[smolvla-trace] register_step_post_hook failed: {e}", flush=True)
            return optimizer, scheduler

        self._patch_attribute(opt_factory, "make_optimizer_and_scheduler", wrapped_make_optimizer)

    def _install_make_policy_hook(self) -> None:
        """Patch lerobot.policies.make_policy 抓住 policy 实例 + dataset meta。"""
        try:
            from lerobot import policies as lp
        except ImportError:
            if self.verbose:
                print("[smolvla-trace] lerobot.policies not importable yet; skipping policy hook")
            return
        original = lp.make_policy
        tracer = self

        def wrapped_make_policy(*args, **kwargs):
            policy = original(*args, **kwargs)
            tracer.captured_policy = policy
            ds_meta = kwargs.get("ds_meta") or (args[1] if len(args) >= 2 else None)
            if ds_meta is not None:
                try:
                    tracer.captured_dataset_meta = {
                        "features": getattr(ds_meta, "features", None) and {
                            k: {
                                "dtype": str(getattr(v, "dtype", "?")),
                                "shape": list(getattr(v, "shape", []) or []),
                            }
                            for k, v in ds_meta.features.items()
                        },
                        "fps": getattr(ds_meta, "fps", None),
                        "total_episodes": getattr(ds_meta, "total_episodes", None),
                        "total_frames": getattr(ds_meta, "total_frames", None),
                    }
                except Exception as e:
                    tracer.captured_dataset_meta = {"error": str(e)}
            if tracer.verbose:
                print(f"[smolvla-trace] captured policy: {type(policy).__name__}", flush=True)
            return policy

        self._patch_attribute(lp, "make_policy", wrapped_make_policy)

    def _install_dataloader_hook(self) -> None:
        """Patch torch.utils.data.DataLoader.__iter__ 抓住第一个 batch 的结构。"""
        from torch.utils.data import DataLoader
        original_iter = DataLoader.__iter__
        tracer = self

        def wrapped_iter(self):
            it = original_iter(self)
            if tracer.captured_first_batch is not None:
                return it

            class _PeekIter:
                def __init__(self, inner):
                    self._inner = inner

                def __iter__(self):
                    return self

                def __next__(self):
                    batch = next(self._inner)
                    if tracer.captured_first_batch is None:
                        tracer.captured_first_batch = _encode_value(batch)
                        if tracer.verbose:
                            print("[smolvla-trace] captured first batch structure", flush=True)
                    return batch

            return _PeekIter(it)

        self._patch_attribute(DataLoader, "__iter__", wrapped_iter)

    # ------------------------------------------------------------------
    # install / uninstall
    # ------------------------------------------------------------------
    def install(self) -> None:
        # global module hooks
        from torch.nn.modules.module import (
            register_module_forward_hook,
            register_module_forward_pre_hook,
        )
        self._handles.append(register_module_forward_pre_hook(self._forward_pre_hook))
        self._handles.append(register_module_forward_hook(self._forward_post_hook))

        # backward 走 tensor.register_hook 走 _forward_post_hook 里挂的, 不再用 module-level
        # full_backward_hook（跟 SmolVLA 的 inplace op 冲突）。

        self._install_functional_hooks()
        self._install_optimizer_sentinel()
        self._install_make_policy_hook()
        self._install_dataloader_hook()

        if self.verbose:
            print(f"[smolvla-trace] hooks installed (forward/backward/functional/optimizer/dataloader)", flush=True)

    def uninstall(self) -> None:
        for h in self._handles:
            with contextlib.suppress(Exception):
                h.remove()
        self._handles.clear()
        for owner, name, original in reversed(self._patched_originals):
            with contextlib.suppress(Exception):
                setattr(owner, name, original)
        self._patched_originals.clear()

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def build_module_tree(self) -> dict:
        """从 captured_policy 走 named_modules 建一棵语义化的 module 树。

        返回 dict: { module_id: { "name": ..., "qualname": ..., "src": ..., "parent_id": ... } }
        """
        if self.captured_policy is None:
            return {}
        tree: dict[int, dict] = {}
        # collect id(parent) for each named module by walking children
        name_to_id: dict[str, int] = {}
        for name, m in self.captured_policy.named_modules():
            name_to_id[name] = id(m)
            src_file, src_line = self._src_loc(m)
            tree[id(m)] = {
                "name": name or "<root>",
                "qualname": f"{type(m).__module__}.{type(m).__name__}",
                "src_file": src_file,
                "src_line": src_line,
                "parent_id": None,
                "num_params": sum(p.numel() for p in m.parameters(recurse=False)),
                "num_params_total": sum(p.numel() for p in m.parameters()),
            }
        # parent linking
        for name, mid in name_to_id.items():
            if not name:
                continue
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent_id = name_to_id.get(parent_name)
            tree[mid]["parent_id"] = parent_id
        return tree

    def dump(self, path: Path | None = None) -> Path:
        if path is None:
            path = self.output_dir / f"trace_{int(time.time())}.json"
        payload = {
            "version": 1,
            "produced_at": time.time(),
            "config": {
                "max_optimizer_steps": self.max_optimizer_steps,
                "capture_backward": self.capture_backward,
                "capture_functional": self.capture_functional,
            },
            "optimizer_step_count": self.optimizer_step_count,
            "dataset_meta": self.captured_dataset_meta,
            "first_batch": self.captured_first_batch,
            "module_tree": self.build_module_tree(),
            "module_events": [e.__dict__ for e in self.module_events],
            "functional_events": [e.__dict__ for e in self.functional_events],
        }
        with open(path, "w") as f:
            json.dump(payload, f, default=str)
        if self.verbose:
            n_fwd = sum(1 for e in self.module_events if e.phase == "forward_post")
            n_bwd = sum(1 for e in self.module_events if e.phase == "backward")
            n_fn = len(self.functional_events)
            print(
                f"[smolvla-trace] wrote {path} "
                f"({n_fwd} forwards, {n_bwd} backward grads, {n_fn} functional ops)",
                flush=True,
            )
        return path


# ---------------------------------------------------------------------------
# Convenience: global tracer singleton (the wrapper uses it)
# ---------------------------------------------------------------------------
_active: Tracer | None = None


def install(output_dir: str | Path, **kwargs) -> Tracer:
    global _active
    if _active is not None:
        raise RuntimeError("Tracer already installed")
    _active = Tracer(Path(output_dir), **kwargs)
    _active.install()
    return _active


def get_active() -> Tracer | None:
    return _active
