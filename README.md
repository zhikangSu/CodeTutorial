# Code Tutorial · Claude Code plugin

Trace one training step of **any PyTorch repo** (forward + backward + 1× optimizer.step), then render the result as an **interactive single-file HTML architecture diagram** — Y-shape topology, composite drill-in, leaf detail drawer with KaTeX formulas and `vscode://` source jumps, and an embedded BYOK "Ask AI" widget.

Ships with a fully worked **SmolVLA** reference example you can study or copy from.

## Install

This repo is its own Claude Code [marketplace](https://docs.claude.com/en/docs/claude-code/plugins). In Claude Code, run:

```
/plugin marketplace add https://github.com/zhikangSu/CodeTutorial
```

Then install the plugin:

```
/plugin install code-tutorial@code-tutorial
```

That's it. The skill is now discoverable. Invoke it by mentioning "trace this training" / "可视化数据流" / "架构图" in conversation, or call it directly:

```
/code-tutorial
```

## What it does

Two stages, both single-file:

### Stage 1 · `instrument.py` — trace one training step

`Tracer` attaches hooks to every `nn.Module` + monkey-patches selected `torch.nn.functional` ops + intercepts `Optimizer.step` via `register_step_post_hook`. After one full forward + backward + `optimizer.step`, it raises `TraceComplete` and dumps a JSON of every call with real tensor shapes, gradient flow, parameter counts, and source locations.

For lerobot users, `trace_runner.py` wraps this around `lerobot-train` automatically. For any other repo, drop `Tracer` into your training script directly (see [`skills/code-tutorial/SKILL.md`](skills/code-tutorial/SKILL.md) for the API).

### Stage 2 · `render_html.py` — turn JSON into clickable architecture

```bash
python skills/code-tutorial/scripts/render_html.py outputs/traces/<repo>/trace_<ts>.json
```

Outputs a self-contained ~170-200 KB HTML file with:

- **Y-shape pipeline** with composite drill-in and leaf detail drawer
- **Real shapes with semantic dim names** — `[batch=16, seq=177, hidden=960]`, never bare numbers
- **Repeated layers collapsed** — 16 transformer blocks → single node + `×16` badge
- **Training-only branches** dashed + low-saturation
- **Backward path toggle** that cuts off at frozen layers (top-right button "↶ 显示反向梯度")
- **Trained / frozen badges** in leaf drawers — derived from backward event counts in the trace
- **Pre-rendered KaTeX formulas** — first drawer open is instant, not 50-100ms-laggy
- **`vscode://` links** that jump to the actual `nn.Module` declaration (not torch internals)
- **Embedded Ask AI widget** — select any text → floating button → side panel, BYOK (OpenAI / Gemini / DeepSeek / Ollama / custom), streaming, thinking-stream support

## Adapting to a new PyTorch repo

The SmolVLA `TREE` is shipped as a **worked example**. For any other repo:

```bash
# 1) Mechanical scaffold — fills the easy stuff, leaves TODO: markers for the rest
python -m render.infer_tree outputs/traces/<repo>/trace_<ts>.json --out-dir inferred/

# 2) Read inferred/digest.md (auto-generated next-step checklist + repo summary)
# 3) Use the prompt template in skills/code-tutorial/references/topology-prompt.md
#    to fill the TODO: fields (concept names, what/blurb, formulas, Y-shape stages)
# 4) Save as render/tree_<your-repo>.py and switch the import in render/__init__.py
```

The full Step 1-8 guide is in [`skills/code-tutorial/references/topology-inference.md`](skills/code-tutorial/references/topology-inference.md).

## Repo layout

```
CodeTutorial/                         # this repo = a single-plugin marketplace
├── .claude-plugin/
│   ├── plugin.json                   # plugin manifest
│   └── marketplace.json              # marketplace catalog (one plugin)
└── skills/
    └── code-tutorial/                # the skill
        ├── SKILL.md                  # workflow Claude reads when triggered
        ├── scripts/
        │   ├── instrument.py         # Tracer class (generic)
        │   ├── trace_runner.py       # lerobot-train wrapper (example runner)
        │   ├── render_html.py        # entry point
        │   └── render/               # render package
        │       ├── tree_smolvla.py   # SmolVLA reference TREE
        │       ├── infer_tree.py     # scaffold tool for new repos
        │       ├── ...
        │       └── assets/           # real .css / .js / .html (not Python r-strings)
        └── references/
            ├── topology-inference.md # ★ how to recover topology from forward + paper
            ├── topology-prompt.md    # ★ paste-able prompt for filling TREE TODOs
            ├── design-system.md
            ├── hook-patterns.md
            └── explanation-style.md
```

## Design principles

- **Single-GPU only** — multi-GPU `accelerate` / DDP pollutes the module tree
- **No source modification** — all capture is via `register_module_forward_hook`, `Optimizer.register_step_post_hook`, tensor-level `register_hook`, and `functools.wraps` monkey-patches
- **Training-time forces offline + no proxy** — model weights must already be in the local HF cache
- **Backward via tensor-level `register_hook`** — `register_full_backward_hook` conflicts with inplace ops common in transformer code

## License

MIT
