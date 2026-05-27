"""trace_runner.py — 在用户机器上跑一次 lerobot-train 并捕获 trace。

用法（典型）：
    cd /data/users/szk/smolvla_project
    CUDA_VISIBLE_DEVICES=0 python .claude/skills/smolvla-trace/scripts/trace_runner.py \\
        --output-dir outputs/traces/v18_single_gpu \\
        --policy.path=lerobot/smolvla_base \\
        --dataset.repo_id=local/so101_cube_into_cup_103ep \\
        --dataset.root=$PROJECT_ROOT/data/SO101/so101_cube_into_cup_103ep \\
        --policy.device=cuda --batch_size=2 --steps=1 \\
        --rename_map='{"observation.images.fixed": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

设计：
1. 用 argparse 解析自己的 --output-dir / --max-steps / --no-backward / --no-functional 等控制参数
2. 剩余 argv 全部转交给 lerobot-train（按 lerobot.scripts.lerobot_train:main 的 parser.wrap 语义）
3. install Tracer（注册全局 module hook + 各种 patch）
4. 调用 lerobot_train.main()
5. 等 TraceComplete 抛出来 → 调用 tracer.dump() → 打印 JSON 路径

关键约束：
- 不修改 lerobot 源码
- 单 GPU only（多 GPU/DDP 留 v2）
- 训练禁代理 + offline 跟 CLAUDE.md 项目准则一致
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 确保 instrument.py 能 import（同目录）
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _set_offline_env() -> None:
    """CLAUDE.md 项目准则：训练时禁代理 + 强制 offline + 用项目本地 HF cache。"""
    for k in (
        "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
        "all_proxy", "ALL_PROXY",
    ):
        os.environ.pop(k, None)
    os.environ.setdefault("no_proxy", "*")
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # 指向项目 HF cache（smolvla_base 的权重在这里）—— 跟训练脚本完全一致
    project_root = os.environ.get("PROJECT_ROOT") or "/data/users/szk/smolvla_project"
    os.environ.setdefault("PROJECT_ROOT", project_root)
    os.environ.setdefault("HF_HOME", f"{project_root}/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", f"{project_root}/cache/huggingface/datasets")
    os.environ.setdefault("TORCH_HOME", f"{project_root}/cache/torch")

    # RTX 4090 P2P/IB workaround（即便单卡也无害）
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")


def _split_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """从混合 argv 里抠出我们自己的开关。

    我们自己的参数都带 `--trace-` 前缀, 避开 lerobot 的 parser.wrap 命名空间。
    """
    parser = argparse.ArgumentParser(
        description="Trace one optimizer step of a lerobot-train run.",
        add_help=False,
    )
    parser.add_argument("--trace-output-dir", required=True, type=Path,
                        help="目录, JSON 写到这里")
    parser.add_argument("--trace-max-steps", default=1, type=int,
                        help="跑几个 optimizer.step 后停（默认 1）")
    parser.add_argument("--trace-backward", action="store_true",
                        help="抓 backward hook（默认关：SmolVLA 含 inplace ops 跟 backward hook 的"
                             " view safety check 冲突，会让 forward 直接报 RuntimeError）")
    parser.add_argument("--trace-no-functional", action="store_true",
                        help="不 wrap functional ops（如果撞 dynamo / torch.compile 问题再开）")
    parser.add_argument("--trace-quiet", action="store_true",
                        help="不打 progress")
    parser.add_argument("-h", "--help", action="store_true")

    known, rest = parser.parse_known_args(argv)
    return known, rest


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    known, lerobot_argv = _split_args(argv)

    if known.help:
        print(__doc__)
        return 0

    _set_offline_env()

    import instrument  # 注意：插入了 scripts/ 到 sys.path

    tracer = instrument.install(
        output_dir=known.trace_output_dir,
        max_optimizer_steps=known.trace_max_steps,
        capture_backward=known.trace_backward,
        capture_functional=not known.trace_no_functional,
        verbose=not known.trace_quiet,
    )

    # 把剩余 argv 交给 lerobot-train 的 parser.wrap
    # lerobot 的 parser.wrap 读 sys.argv[1:], 所以重写 sys.argv
    sys.argv = ["lerobot-train", *lerobot_argv]

    if not known.trace_quiet:
        print(f"[smolvla-trace] handing argv to lerobot-train:", flush=True)
        for a in lerobot_argv:
            print(f"  {a}", flush=True)

    rc = 0
    try:
        from lerobot.scripts.lerobot_train import main as lerobot_main
        lerobot_main()
    except instrument.TraceComplete as e:
        if not known.trace_quiet:
            print(f"[smolvla-trace] TraceComplete: {e}", flush=True)
    except SystemExit as e:
        # 某些 parser.wrap 实现会 sys.exit, 当作正常结束
        if not known.trace_quiet:
            print(f"[smolvla-trace] caught SystemExit({e.code}), continuing to dump", flush=True)
        rc = int(e.code or 0)
    except Exception as e:
        # 让用户能看到训练侧真正的异常, 同时仍然 dump 出已收集的 trace
        import traceback
        print(f"[smolvla-trace] training raised {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        rc = 1
    finally:
        out_path = tracer.dump()
        tracer.uninstall()
        if not known.trace_quiet:
            print(f"[smolvla-trace] DONE → {out_path}", flush=True)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
