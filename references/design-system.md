# Design system

> Codebase-to-course 派系的暖 off-white + vermillion + Bricolage Grotesque。改 `render_html.py` 里的 `CSS` 常量或 explainer 时按这套用。

## Color tokens

CSS variables 全部在 `:root`：

| token              | hex       | 用途                               |
|--------------------|-----------|------------------------------------|
| `--bg`             | `#faf6f0` | 页面底色 (warm off-white)          |
| `--bg-card`        | `#fffdf9` | 卡片底色                           |
| `--bg-card-hover`  | `#fff7ec` | 卡片 hover                         |
| `--bg-explainer`   | `#fff2e3` | explainer 块底色 (淡橙)            |
| `--ink`            | `#1a1612` | 正文文字                           |
| `--ink-soft`       | `#5a4f43` | 次要文字 / 灰说明                  |
| `--rule`           | `#e7dccc` | 分割线                             |
| `--accent`         | `#d94f1e` | vermillion 主强调色                |
| `--accent-soft`    | `#f4d5c4` | 强调色 5% 淡背景                   |
| `--accent-deep`    | `#a13912` | 强调色深色（标题/链接）            |
| `--code-bg`        | `#1a1612` | 代码块底色 (深棕)                  |
| `--code-ink`       | `#f4e9d8` | 代码块文字                         |
| `--dim-tag`        | `#a06b3e` | 维度标签（b/c/h/w）的金棕色        |

避免：纯白 (`#fff`) 当 bg、紫色渐变、Inter/Roboto/Arial 字体——这些都是 frontend-slides skill 里写的 "AI slop" 反例。

## Typography

| family                | weight     | 用途                  |
|-----------------------|------------|-----------------------|
| Bricolage Grotesque   | 400/500/600 | display (h1, h2, h3)  |
| DM Sans               | 400/500/600 | body text             |
| JetBrains Mono        | 400/500    | code, tensor pills, src refs |

通过 Google Fonts CDN：

```html
<link href="https://fonts.googleapis.com/css2?
family=Bricolage+Grotesque:wght@400;500;600&
family=DM+Sans:wght@400;500;600&
family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

## Layout

- `main` width: 1020px max, centered, padding 32px
- `scroll-snap-type: y proximity`（**不要 mandatory**，长 phase 会被卡死）
- `section.phase` 顶部一根 `1px solid var(--rule)` 分割线 + `padding-top: 10px` 留呼吸
- explainer 卡用 `border-left: 3px solid var(--accent)` 做"重点标记条"

## 维度标注 (核心约定)

不要写裸 `[2, 197, 64]`。一律按位置语义化：

```
b=2 · seq=197 · hidden=64
```

通过 `<span class='dim'>` (golden) + `<span class='dim-lit'>` (ink) 着色。`annotate_shape()` 根据 module qualname + shape 长度切换标签：

| 模式                     | 启发式 (in `annotate_shape`)                    |
|--------------------------|--------------------------------------------------|
| 4D 图像/feature          | b · c · h · w                                    |
| 3D action chunk          | b · chunk · action_dim（hint=action）            |
| 3D vision / attention    | b · seq · hidden（"Vision"/"Attention"/"Llama" in name）|
| 3D 通用                  | b · t · d                                        |
| 2D Linear input          | b·seq · d                                        |
| 2D 通用                  | b · d                                            |

## 数学公式

KaTeX 0.16 from jsdelivr CDN（已内嵌 `<script defer>` + auto-render 配置）。

- 显示模式：`$$ ... $$`
- 行内模式：`\\( ... \\)`（注意 Python raw string 里写 `\(`）

Python 端 explainer 公式字符串用 raw string + `r"$$...$$"`，避免反斜杠转义噩梦。

不联网时 KaTeX JS 不会加载，公式会以原始 LaTeX 文本回退显示——不影响理解，但难看；如果用户离线场景多可以考虑后续把 KaTeX 打包成内联资源。

## 反例（不要做）

- 在 explainer 里手写"大白话讲解"——v0 决策是只引博客原话。要扩 explainer 先去 HF blog / 第三方 flow matching 博客找 quote。详见 [explanation-style.md](explanation-style.md)。
- 用 `position: absolute` / 父容器 `overflow: hidden` 的 tooltip——会被裁掉。如果以后加 tooltip 走 `position: fixed` + `getBoundingClientRect()`，append 到 `document.body`。
- 给模块卡加 fancy 渐变。维持平铺单色面 + 1px border 的"editorial"质感，跟内容（论文+公式+真实 shape）的密度匹配。
