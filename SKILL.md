---
name: smolvla-trace
description: 跑一次目标 repo 的一个训练 step (forward + backward + optimizer.step)，捕获所有 nn.Module forward call + 真实 tensor shape + backward grad，按论文+源码推出顶层架构图，渲染成可交互单文件 HTML（Y/fork-join 拓扑、composite 下钻、leaf 详情抽屉、维度语义化标注、KaTeX 公式、vscode:// 源码跳转）。SmolVLA 已是参考实现；遇到新 repo 时按 references/topology-inference.md 的 Step 1-8 推 TREE 后再渲染。
---

# Trace skill — 从训练命令到可交互架构图

针对一个 PyTorch 训练命令（默认 SmolVLA，也能扩到其他 repo），"运行一遍 → 抓所有 forward + backward → 推顶层拓扑 → 渲交互 HTML"。

## 什么时候触发

用户提"trace / 溯源 / 链路 / pipeline 可视化 / 数据流 / 架构图 / 怎么算的"等关键词时。

不限制 SmolVLA：第一步会问用户目标 repo 和有没有论文，按 [references/topology-inference.md](references/topology-inference.md) 推 TREE。SmolVLA 的 TREE 已经预置在 `scripts/render_html.py` 里作为参考实现。

不要用在：
- 推理路径分析（只 hook 训练前向 + 一次 optimizer.step）
- 多卡训练（accelerate/DDP 会污染模块树，必须改成单 GPU）

## 工作流（按顺序）

### Step 0 · 问用户

必问的三件事（如果用户没明说就一次性问全）：

1. **训练命令**：完整的 `lerobot-train ...` 或等价命令（多 GPU 改成单 GPU）
2. **目标 policy 的源码位置**：`lerobot/policies/smolvla/` 这种本地路径
3. **有没有论文 / README / overview 图**：有的话 paste 路径（PDF 转 md / arXiv 链接 / 项目根的 README），用于 [topology-inference.md](references/topology-inference.md) 的 Step 3、Step 8

如果用户说"我就要默认 SmolVLA 版"，直接跳过 Step 0 用预置 TREE。

### Step 1 · 跑 trace

**原则**：除 `--steps` 外参数尽量跟原训练命令一致（包括 `--batch_size`）。多 GPU -> 单 GPU。

SmolVLA 默认命令：

```bash
ACCELERATE_MIXED_PRECISION=bf16 \
CUDA_VISIBLE_DEVICES=0 python .claude/skills/smolvla-trace/scripts/trace_runner.py \
  --trace-output-dir=outputs/traces/v18_demo \
  --trace-max-steps=1 \
  --trace-backward \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/so101_cube_into_cup_103ep \
  --dataset.root=/data/users/szk/smolvla_project/data/SO101/so101_cube_into_cup_103ep \
  --policy.device=cuda \
  --batch_size=16 \
  --steps=2 \
  --output_dir=outputs/checkpoints/_trace_scratch \
  --job_name=trace_demo \
  --save_freq=10 --log_freq=1 \
  --wandb.enable=false \
  --dataset.video_backend=pyav \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.fixed": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

关键改动 vs 原 v18 命令：
- `accelerate launch --multi_gpu --num_processes=4` -> 单 GPU 直接 python 调
- `--steps=30000` -> `--steps=2`（sentinel 在 step #1 完成后中断, step 2 不会真跑）
- `--mixed_precision=bf16` -> `ACCELERATE_MIXED_PRECISION=bf16` 环境变量（Accelerator 自动读）
- 其他全部保持原样（`batch_size=16`, dataset, rename_map）

输出 `outputs/traces/<name>/trace_<unix_ts>.json`，~1-2 MB。

### Step 2 · 推 TREE（新 repo 才需要）

SmolVLA 跳过这步 —— TREE 已经预置在 `scripts/render/tree_smolvla.py` 里。

新 repo 时，先用 `infer_tree.py` 跑骨架，再用 [references/topology-inference.md](references/topology-inference.md) 补语义：

```bash
# 1) 机械化推骨架（path_map / decl_src / tree_skeleton / digest.md）
cd scripts
python -m render.infer_tree outputs/traces/<repo>/trace_<ts>.json --out-dir inferred/
```

输出 4 个文件:
- `inferred/path_map.py`     — 自动检测的 `PATH_MAP`（src_file 前缀映射）
- `inferred/decl_src.py`     — AST 找到的 leaf 声明位置（替换 trace 的 torch 内部 forward 行）
- `inferred/tree_skeleton.py`— TREE 骨架（trace_name / src / repeat 全填好；name / sub / what / blurb 标 `TODO:`）
- `inferred/digest.md`       — 给 Claude 的可读摘要 + 下一步 checklist

```bash
# 2) Claude 在对话里读 digest.md + 论文/README, 按 topology-inference.md 改 tree_skeleton.py
#    - 把 TODO: 字段换成真正的 concept_name / blurb / what
#    - 把 pipeline.stages 从 linear 改成 Y-shape (按 forward 语法顺序 + 论文)
#    - 标 is_heart / training_only
#    - 给关键 leaf 补 formula / callout

# 3) 改好后改名为 render/tree_<repo>.py, 在 render/__init__.py 里替换 TREE 的 import 来源
```

infer_tree.py 只做机械化部分（trace 真相 + AST 解析），不试图推语义。SmolVLA-specific 硬编码集中在
`render/tree_smolvla.py`、`render/source_links.py` 两个文件，按 inferred 的输出替换即可。

### Step 3 · 渲染 HTML

```bash
python .claude/skills/smolvla-trace/scripts/render_html.py outputs/traces/v18_demo/trace_<ts>.json
```

输出 `outputs/traces/v18_demo/trace_<ts>.html`（同目录同名）。

特性：
- **Y-shape 拓扑**：raw_batch fork 到 embed_prefix / embed_suffix 双分支，到 vlm_expert 合并
- **可交互**：composite 节点点击下钻、leaf 节点点击右抽屉详情、Esc 回退、breadcrumb 任意一段可点
- **vscode://** 源码链接：点击 src 直接跳 VSCode 对应文件 + 行号
- **语义化维度**：`[batch=16, seq=177, hidden=960]`，不出现裸数字
- **重复层合并**：16 层 VLM body 渲成单节点 + ×16 badge（不会刷出 16 个相同方块）
- **训练-only 虚线**：flow_loss / backward_step 虚线 + 灰底
- **KaTeX 公式**：行内 `\(x_t\)` 和块级 `$$...$$` 都渲染（需联网取 KaTeX CDN）

### Step 4 · 用户反馈调整

常见调整：
- **想改某个 leaf 的 what 段**：编辑 `scripts/render_html.py` 里对应 TREE 节点的 `what` 字段
- **想加新顶层方块**：在 `pipeline.stages` 里加一个，再写对应节点
- **维度标注不对**：编辑 `_dim_names()` 启发式（按 shape 长度 + hint）

## 设计约束 / 已知限制

1. **不修改目标 repo 源码**。所有捕获走 `torch.nn.modules.module.register_module_forward_hook`、`Optimizer.register_step_post_hook`、tensor-level `register_hook`、`functools.wraps` monkey-patch。Skill 目录 (`scripts/`) 是唯一新增的代码。
2. **单 GPU only**。多 GPU 下 accelerate 会包 DDP wrapper，模块树会多 `module.` 前缀，hook 也会在所有 worker 进程上同时开。
3. **训练时强制 offline + 禁代理**（CLAUDE.md 项目规则）。`trace_runner.py` 自动 `unset http_proxy* + export HF_HUB_OFFLINE=1`。所以模型权重和数据集都必须已经在项目 HF cache 里。
4. **第一次 forward 才捕获**。后续 step 的 forward 不会被 hook（在第一次 `optimizer.step` 后 raise `TraceComplete` 中断了训练）。
5. **functional ops 默认只 wrap `F.scaled_dot_product_attention`**。其他 functional（`F.silu`、`F.gelu`、`F.mse_loss` 等）没 wrap，因为它们要么少要么走 nn.Linear 已被 hook。如果新 repo 用到别的 functional 想看，往 `instrument._install_functional_hooks()` 加 wrap。
6. **dataset/collate 抓"第一个 batch"的结构而不是值**。`_encode_value` 只记 tensor shape/dtype，不存张量值——HTML 永远不会泄露真实图像 / 动作数据。
7. **backward 通过 tensor-level register_hook**（不走 module-level `register_full_backward_hook`）—— 后者跟 SmolVLA 的 inplace op 冲突。

## 文件清单

```
.claude/skills/smolvla-trace/
├── SKILL.md                       # 本文件
├── scripts/
│   ├── instrument.py              # Tracer 类：forward/backward/functional/optimizer hook + JSON dump
│   ├── trace_runner.py            # CLI wrapper：env 设置 + 解析 --trace-* 自参数 + 调 lerobot-train.main()
│   ├── render_html.py             # 入口：trace JSON + TREE -> 单文件 HTML
│   └── render/                    # 拆分后的渲染包
│       ├── __init__.py
│       ├── trace_utils.py         # load_trace / fmt_shape / pair_events
│       ├── source_links.py        # PATH_MAP / SMOLVLA_DECL_SRC / vscode_link
│       ├── tree_smolvla.py        # ★ SmolVLA-specific TREE（换 repo 时替换为 tree_<repo>.py）
│       ├── enrich.py              # 把 trace 真实 shape 填进 TREE
│       ├── assets.py              # 资源 loader
│       ├── infer_tree.py          # ★ 通用化工具：trace JSON -> TREE 骨架 + digest.md
│       └── assets/                # 真文件 CSS / JS / HTML（不再是 Python 字符串）
│           ├── main.css, main.js, katex.html
│           ├── ask_ai.css, ask_ai.html
│           └── ask_ai_js/         # IIFE 按职责拆 12 个文件, 字典序拼接
└── references/
    ├── topology-inference.md      # ★ 给 Claude 看的：怎么从论文+源码+trace 推顶层拓扑（Step 1-8）
    ├── design-system.md           # 配色 / 字体 / 卡片 design tokens
    ├── hook-patterns.md           # PyTorch hook 边界情况：functional op wrap / optimizer post_hook / tensor backward hook
    └── explanation-style.md       # 节点 narrative 写法约定
```

## 经验/教训（实现里踩过的坑）

1. **`Optimizer.step` 不能 patch 在基类**。AdamW 等子类有自己的 `step` 覆盖。改用 `optimizer.register_step_post_hook(...)`（torch ≥ 1.13 官方 API），patch 在 `make_optimizer_and_scheduler` 返回时。
2. **backward hook 跟 inplace op 冲突**。SmolVLMWithExpertModel 用 `out_emb += hidden_states`，注册 `register_full_backward_hook` 后 PyTorch view safety check 会让这条 inplace 报 RuntimeError。改成 `tensor.register_hook(...)` 直接挂在张量上，不插 BackwardHookFunction wrapper，没这个问题。
3. **顶层 forward chain 多层直接调 `.forward()`**。`SmolVLAPolicy.forward` 里 `self.model.forward(...)`、`VLAFlowMatching.forward` 里 `self.vlm_with_expert.forward(...)` 都绕过 `__call__`，所以那些 wrapper module 不会触发 forward hook。结果是 leaf module（Linear / RMSNorm / MLP）全都出现在 depth=0。**这正是 topology-inference Step 1 强调的"trace 顺序不等于 forward 语义顺序"的来源**。
4. **HF_HOME 必须显式指**。lerobot-train 没自动用项目 HF cache，所以 trace_runner 必须 `os.environ.setdefault("HF_HOME", f"{PROJECT_ROOT}/cache/huggingface")`，否则会撞 offline mode 找不到 smolvla_base。
5. **lerobot 启动会 raise FileExistsError**：output_dir 已存在且非 resume 模式。trace 之间要清 scratch dir（`rm -rf outputs/checkpoints/_trace_scratch`）或者 job_name 带 timestamp。
