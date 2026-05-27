# Explainer 写法约定

> v0 决策：**explainer 大白话不自己编**，引博客原话 + 让 LLM (你自己) 仅做最后的"串接"。这是用户 2026-05-22 凌晨明确给的方向。

## 三件套结构

每条 explainer 是 `EXPLAINERS[<key>]` 字典，包含三个字段：

| 字段                  | 内容                                                          |
|-----------------------|---------------------------------------------------------------|
| `title`               | 大标题（中文，带 `·` 分隔副标题）                              |
| `narrative_html`      | 一段或几段 HTML，**主体是 blockquote 引文**，串接句子写在引文之间 |
| `formula_html`        | LaTeX 字符串（KaTeX 渲），用 raw string `r"$$...$$"` 避免转义  |
| `shape_change_html`   | shape 流动（`<div class='shape-flow'>` 单调字体块，不走 LaTeX）|

## Narrative 必走 blockquote

```python
"narrative_html": (
    "<p>简短中文铺垫，提示这一步要做什么。</p>"
    + quote(
        "Image tokens are extracted via the vision encoder. Language instructions "
        "are tokenized and fed directly into the decoder. ...",   # ← 原话, 不改
        "HF SmolVLA Blog",                                          # ← 来源 label
        "https://huggingface.co/blog/smolvla")                      # ← URL
    + "<p>串接句：解释引文跟当前 trace 里看到的具体 module/shape 怎么对应。</p>"
)
```

`quote(text, label, url)` helper 在 `render_html.py` 顶部定义。

## 引文来源（v0 用过的）

| 来源                              | URL                                                        | 适合写哪些 phase                    |
|-----------------------------------|------------------------------------------------------------|-------------------------------------|
| HF SmolVLA Blog (Aug 2025)        | https://huggingface.co/blog/smolvla                        | 所有 SmolVLA 架构相关：VLM / connector / action expert / flow matching |
| Federico Sarrocco · Flow Matching | https://federicosarrocco.com/blog/flow-matching            | flow matching 在机器人控制里的直观比喻 |
| Cambridge MLG · Flow Matching     | https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html | flow matching 数学定义、跟 CNF 关系  |

加新引文时记得 attribute 到原作者，并把链接放到来源 label 里（HTML render 会变成可点击 cite）。

## 公式约定

- 用 LaTeX，KaTeX 渲。`raw string r"$$...$$"` 写。
- 单条公式独占一行；多条公式之间不留空白行（auto-render 会按 $$ 分块）。
- 用 `\boxed{...}` 突出最终 loss/objective（如 flow matching ℒ = E ‖v_t − u_t‖²）。
- 注释行可以放在公式块底部的小 `<p>`（非 LaTeX），用 `font-size:11px; color:#5a4f43;`。

## Shape 流动

- 用 `<div class='shape-flow'>` 单调字体块。
- 形状用 `ℝ<sup>b · 50 · 32</sup>` 排版（unicode 数学黑板体 + sup），不用 LaTeX——快、不需要 KaTeX。
- 标真实 trace 里的尺寸（这次 demo 是 b=2, chunk=50, action_dim=6, max_action_dim=32）。

## 串接句的边界

引文不会跟用户当前的具体 trace 完全对齐（HF blog 描述的是通用 SmolVLA，trace 是 SO101 cube_into_cup_103ep）。串接句的职责：

1. **指明 trace 里哪个 module 对应这段引文**：例如 "trace 看到 SmolVLMVisionTransformer 被调用 2 次（两路相机各一次）"。
2. **挑出非 trivial 的具体细节**：例如 "训练日志能看到 'Reducing the number of VLM layers to 16'——就是上面这条 half the total layers 的代码体现"。
3. **解释 trace 表面奇怪现象**：例如 "这个 phase 在 trace 里 depth=0 调用全是 Linear/RMSNorm——因为 SmolVLMWithExpertModel 用 `.forward()` 直接调子模块（绕过 `__call__`）"。

不要让串接句自己充当"大白话讲解"——那是引文的活。串接句只做"把博客 → trace"的桥接。

## 加新 explainer 的清单

1. 在 `EXPLAINERS` 加 key + dict（`title` / `narrative_html` / `formula_html` / `shape_change_html`）。
2. 在相应 phase 切分逻辑 (`split_phases` / `_phase_key`) 让某个 phase 的 `explainer_key` 指到这个新 key。
3. 至少塞 2 条 blockquote 引文。不够就**别加这个 explainer**，回去找更多博客。
4. 数学公式用 LaTeX，shape 流用 unicode + `<sup>`。
5. 不写"经验之谈"——经验之谈写到 SKILL.md 或者 memory 里。

## 为什么这么严

用户的原话（2026-05-22 凌晨）：

> 大白话你可以不用自己编，你可以看看网上的一些博客，看看他们是怎么讲这些的

LLM 编"大白话"的失败模式：术语堆砌、平均化、看似准确实则没新信息。而博客是真人写给真人的，已经经过"有没有人愿意读完"的过滤。我们做的是**编辑工作**（挑、串、对齐 trace），不是**写作工作**。
