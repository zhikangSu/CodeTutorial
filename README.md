# SmolVLA Trace · Claude Code Skill

把一条 SmolVLA 训练命令变成一张可交互的架构图 HTML：跑一次 forward + backward，捕获所有 module forward call + 真实 tensor shape + backward grad，按论文 + 源码推出顶层 Y-shape 拓扑，渲染成可点击下钻的单文件 HTML（KaTeX 公式、vscode:// 源码跳转、内嵌 Ask AI BYOK 问答）。

## 这是什么

一个 [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills)，针对 [LeRobot SmolVLA](https://huggingface.co/blog/smolvla) 训练 trace 的可视化工具。也可以推广到其他 PyTorch 训练 repo（参见 [`references/topology-inference.md`](references/topology-inference.md)）。

跑出来的 HTML 长这样：

- 首屏 Y-shape 拓扑图：`raw_batch → embed_prefix / embed_suffix → vlm_expert → action_proj → flow_loss → backward_step`
- 复合节点点击下钻，叶子节点点击右抽屉详情
- 每个 leaf 详情含真实 shape（语义化维度名 `[batch=16, seq=177, hidden=960]`，从 trace 抓的）、公式（KaTeX）、源码 vscode:// 链接
- 嵌入式 "问 AI"：选中任意文字 → 浮动按钮 → 右侧问答面板 → 流式回答 + 思考过程显示。BYOK 支持 OpenAI / Gemini / DeepSeek / Ollama 本地

## 用法

### 1. 装到 Claude Code

```bash
# 把这个仓库 clone 到 Claude Code 的 skills 目录
git clone https://github.com/zhikangSu/CodeTutorial.git ~/.claude/skills/smolvla-trace

# 或者项目级
cd <your-project>
mkdir -p .claude/skills
git clone https://github.com/zhikangSu/CodeTutorial.git .claude/skills/smolvla-trace
```

### 2. 跑 trace

```bash
# 单 GPU 复刻你的训练命令, 只改 --steps=2 + --trace-* 参数
ACCELERATE_MIXED_PRECISION=bf16 \
CUDA_VISIBLE_DEVICES=0 python .claude/skills/smolvla-trace/scripts/trace_runner.py \
  --trace-output-dir=outputs/traces/demo \
  --trace-max-steps=1 \
  --trace-backward \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/<your-dataset> \
  --dataset.root=<dataset-path> \
  --policy.device=cuda --batch_size=16 --steps=2 \
  --output_dir=outputs/checkpoints/_trace_scratch \
  --job_name=trace --save_freq=10 --log_freq=1 \
  --wandb.enable=false --dataset.video_backend=pyav \
  --policy.push_to_hub=false \
  --rename_map='{"observation.images.fixed": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

输出 `outputs/traces/demo/trace_<timestamp>.json` (~1-2MB)。

### 3. 渲染 HTML

```bash
python .claude/skills/smolvla-trace/scripts/render_html.py outputs/traces/demo/trace_<ts>.json
```

输出同目录同名 `.html`（~170KB 单文件，含完整交互逻辑）。浏览器打开即可。

### 4. 用 Ask AI 问答（可选）

打开 HTML 后：
- 选中任意文字 → 浮动 "问 AI" 按钮 → 点击 → 右侧面板
- 第一次会弹出 BYOK 配置（Gemini 免费 key 在 [aistudio.google.com/apikey](https://aistudio.google.com/apikey)）
- 支持流式输出、思考过程（DeepSeek-R1 / Anthropic extended thinking）、停止生成、对话保留轮数

## 文件结构

```
.
├── SKILL.md                       # 工作流文档（Claude Code 读这个）
├── scripts/
│   ├── instrument.py              # Tracer 类：forward/backward/functional hook + JSON dump
│   ├── trace_runner.py            # CLI wrapper：env 设置 + 转发 lerobot-train
│   └── render_html.py             # JSON + TREE → 单文件 HTML
└── references/
    ├── topology-inference.md      # ★ 给 Claude 看的：怎么从论文+源码+trace 推顶层拓扑（Step 1-8）
    ├── design-system.md           # 配色 / 字体 / 卡片 tokens
    ├── hook-patterns.md           # PyTorch hook 边界情况
    └── explanation-style.md       # 节点 narrative 写法约定
```

## 设计要点

- **单 GPU only**（多卡 accelerate/DDP 会污染 module 树）
- **不修改 lerobot/SmolVLA 源码**，所有捕获走 module hook + monkey-patch
- **训练时强制 offline + 禁代理**（HF_HUB_OFFLINE=1），权重必须已在本地 cache
- **tensor-level backward hook**（不走 module 的 full_backward_hook，跟 SmolVLA inplace op 冲突）
- **vscode:// 源码链接**指向 SmolVLA / SmolVLM / Llama 源码声明位置（不是 torch 内部 forward）

## 适配到其他 PyTorch repo

按 [`references/topology-inference.md`](references/topology-inference.md) 的 Step 1-8 推新 repo 的 TREE：读顶层 forward 拿语义顺序、识别拓扑（Y / fork-join / linear）、给每个节点起 concept_name + code_name + purpose_line、shape 用语义化维度名、判定 composite vs leaf、标心脏方块 + 训练-only 分支、写 story_line。

然后替换 `scripts/render_html.py` 顶部的 `TREE` dict 即可。

## License

MIT
