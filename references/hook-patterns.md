# PyTorch hook patterns（trace 实现备忘）

> 这些 pattern 是 `scripts/instrument.py` 里实际用到的。要扩 trace 的覆盖面（更多 functional ops、不同 optimizer、其他 policy）时翻这个。

## 1. 全局 module forward hook

```python
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)

pre = register_module_forward_pre_hook(lambda module, inputs: ...)
post = register_module_forward_hook(lambda module, inputs, outputs: ...)
# 卸载: pre.remove(); post.remove()
```

注意：hook 只在 `module(input)` (`__call__`) 路径上触发；用户代码直接调 `module.forward(input)` 会绕过 hook。SmolVLA 的 forward chain (`policy.forward` → `model.forward` → `vlm_with_expert.forward`) 全是直接 `.forward()`，所以那几层 wrapper 不会出现在 trace 里，**leaf module 看起来全部在 depth=0**。

## 2. Optimizer step sentinel

base class `Optimizer.step` 不能 monkey-patch——AdamW 等子类有自己的 `step` 覆盖，base patch 永远不会被走到。

正确做法：

```python
# 跟 lerobot 的 factory 抓 optimizer 实例
from lerobot.optim import factory as opt_factory
original_make = opt_factory.make_optimizer_and_scheduler

def wrapped_make(*args, **kwargs):
    optimizer, scheduler = original_make(*args, **kwargs)
    optimizer.register_step_post_hook(my_callback)   # ← torch ≥ 1.13 官方 API
    return optimizer, scheduler

opt_factory.make_optimizer_and_scheduler = wrapped_make
```

`register_step_post_hook` 接受 `callback(optimizer, args, kwargs)`，可以在里面抛 `TraceComplete` sentinel 中断训练循环。

## 3. Backward hook（默认关）

```python
from torch.nn.modules.module import (
    register_module_full_backward_hook,
    register_module_full_backward_pre_hook,
)
```

⚠️ 跟 inplace op 冲突。SmolVLA 的 `smolvlm_with_expert.py:529` 有 `out_emb += hidden_states`——这是个 view 上的 inplace 写入。注册 `full_backward_hook` 后 PyTorch 会把 module output 包成 `BackwardHookFunctionBackward`，view safety check 看到 inplace 写入直接报：

```
RuntimeError: Output 0 of BackwardHookFunctionBackward is a view and is being
modified inplace.
```

注意：这个 RuntimeError 是 **forward 时** 报的（不是 backward），因为 view 检测在 forward chain 里就插入了。所以一旦注册 backward hook，**整个 forward 都跑不了**。要开 backward 必须先改 lerobot 源码把 `out_emb += hidden_states` 改成 `out_emb = out_emb + hidden_states`（或 `.clone()` 后再 inplace）。

## 4. Functional op wrap

`F.scaled_dot_product_attention` 不是 nn.Module，hook 抓不到。改 monkey-patch：

```python
import torch.nn.functional as F
original_sdpa = F.scaled_dot_product_attention

def wrapped_sdpa(*args, **kwargs):
    record_event_pre(args, kwargs)
    out = original_sdpa(*args, **kwargs)
    record_event_post(out)
    return out

F.scaled_dot_product_attention = wrapped_sdpa
# 卸载: F.scaled_dot_product_attention = original_sdpa
```

加更多 functional op（`F.linear` / `F.gelu` / `F.softmax` 等）时按这个 pattern。注意性能：每个 op wrap 都加一次 Python lock + dict 写入，过度 wrap 会显著拖慢 trace 速度。

## 5. DataLoader 第一个 batch 抓取

`collate_fn` 不是 nn.Module；要捕获原始 batch 结构得 wrap `DataLoader.__iter__`：

```python
original_iter = DataLoader.__iter__

def wrapped_iter(self):
    it = original_iter(self)

    class PeekIter:
        def __init__(self, inner): self.inner = inner
        def __iter__(self): return self
        def __next__(self):
            batch = next(self.inner)
            if tracer.captured_first_batch is None:
                tracer.captured_first_batch = encode(batch)
            return batch

    return PeekIter(it)

DataLoader.__iter__ = wrapped_iter
```

只抓第一个 batch（足够构图）；后续 batch 不再 encode，避免 trace 文件膨胀。

## 6. make_policy 抓 policy 实例

```python
from lerobot import policies as lp
original = lp.make_policy

def wrapped_make_policy(*args, **kwargs):
    policy = original(*args, **kwargs)
    tracer.captured_policy = policy
    return policy

lp.make_policy = wrapped_make_policy
```

事后用 `policy.named_modules()` 把 trace 里的 `id(module)` 映射回语义化路径（`model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj` 这种）。

## 7. Shape 编码 schema

只记结构, 不记值：

```python
def encode_value(x):
    if isinstance(x, torch.Tensor):
        return {"_type": "Tensor", "shape": list(x.shape),
                "dtype": str(x.dtype), "device": str(x.device)}
    elif isinstance(x, (list, tuple)):
        return {"_type": type(x).__name__,
                "items": [encode_value(v) for v in x[:32]], "len": len(x)}
    elif isinstance(x, dict):
        return {"_type": "dict",
                "items": {k: encode_value(v) for k, v in list(x.items())[:64]},
                "len": len(x)}
    ...
```

`_type` 字段是 renderer 端做 type-dispatch 的 key。改 schema 时记得同时改 `render_value()`。

## 8. Sentinel 异常

`TraceComplete` 是个普通 Exception 子类。在 `register_step_post_hook` 回调里抛出，会传过 `accelerator.backward` → `update_policy` → train loop → 最终在 `trace_runner.py` 的 try/except 里捕获。

trace_runner 的 finally 块永远会调 `tracer.dump()` 和 `tracer.uninstall()`，所以即使训练侧抛非 `TraceComplete` 的真实 exception（比如 dataset 找不到），也能把已采到的部分写盘 + 把所有 hook 卸干净。
