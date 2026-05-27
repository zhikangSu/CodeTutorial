# topology-prompt — 给 Claude 用的"把 TODO 骨架填满"prompt

> 这份不是给人读的文档说明，是一个**可粘贴的 prompt**。Claude 在新 repo 走完 `infer_tree.py` 后，把这份 prompt + `inferred/digest.md` + `inferred/tree_skeleton.py` + 论文/README 一起读，就能产出最终 `tree_<repo>.py`。
>
> 用法（在新对话里）:
> ```
> 把这份 prompt 全文贴进来 + paste inferred/digest.md + paste tree_skeleton.py + paste 论文 abstract
> ```

---

## 任务

你拿到了 `infer_tree.py` 机械化推出来的 TREE 骨架 (`tree_skeleton.py`)，里面所有 `name`/`sub`/`tooltip`/`what`/`blurb`/`story_line` 都标了 `TODO:`。你要按下面 8 步把它们替换成最终内容，让 SVG 渲出来"用户第一眼就懂"。

骨架已经做好的事情（**不要改**）:

- `trace_name`: 模块的 dotted 路径（用来拉真实 shape）
- `src`: 模块声明位置（vscode:// 跳转用）
- `repeat`: 重复层数（如 12 层 ViT block 折叠后 repeat=12）
- `in_hint` / `out_hint`: shape 维度命名提示
- `children`: composite 的子节点 id list
- 整个 module hierarchy

骨架**没**做的事情（你要补）:

- 所有语义字段（concept_name / what / blurb 等）
- `pipeline.stages` 的 Y-shape 拓扑重构（默认 linear，每个 stage 1 node）
- `is_heart` / `training_only` 标记
- 关键 leaf 的 `formula` / `callout`
- `story_line`（顶部 blurb）

---

## Step 1 — 找顶层 forward 的语义顺序

读 `<root>` class（digest.md 第一节有 qualname）的 `forward()` 方法源码。**按语法顺序**抽出每行 `self.X(...)` 调用 + 重要变量赋值（噪声采样、张量插值、loss）。

这个顺序就是 `pipeline.stages` 的**真实**顺序，跟 trace JSON 的事件顺序经常不一样（子模块先 fire forward_pre）。

参考 `references/topology-inference.md` Step 1。

## Step 2 — 识别拓扑（Y / fork-join / linear）

> **⚠️ 不要去 `references/topology-inference.md` 的拓扑词汇表里"找匹配"**。那是 vocabulary 不是 checklist。
> 正确做法是亲自画依赖图，让拓扑从代码里"长出来"——而不是套模板。

从 forward 抽取依赖图：每条 `=` 左边的变量名是 node，右边引用的变量是入边。

- 如果两个变量**独立产生**后被同一调用消费 → **Y-shape merge**
- 如果一个变量被复制走两条路径再合并 → **fork-join**
- 否则就是 **linear**

例（SmolVLA）:
```python
prefix_tokens = self.embed_prefix(images, lang, state)   # 独立
noise, t = self.sample_noise_and_time(...)                # 独立
x_t = (1 - t) * noise + t * action                        # 用 noise
suffix_tokens = self.embed_suffix(x_t, t)                 # 独立链
h = self.vlm_with_expert(prefix_tokens, suffix_tokens)    # 合并 ★ Y 的合并点
```

→ stages:
```python
stages = [
    {"nodes": ["raw_batch"]},
    {"nodes": ["embed_prefix", "embed_suffix"]},      # 并列
    {"nodes": ["vlm_expert"], "is_heart": True},      # merge
    {"nodes": ["action_proj"]},
    {"nodes": ["flow_loss"], "training_only": True},
    {"nodes": ["backward_step"], "training_only": True},
]
```

## Step 3 — 给每个 node 起 concept_name + sub

替换 `name: "TODO: foo"` 为 5 词以内、动词领头的概念名。`sub` 改成 hint 信息（类名 + 关键维度）。

| Code 里叫 | concept_name (`name`) | sub |
|---|---|---|
| `embed_prefix` | 观测编码 | `embed_prefix · vision + lang + state` |
| `action_in_proj` | Action in proj | `Linear 32→720` |
| `flow_loss` | Flow loss | `F.mse_loss(v_t, ε−a)` |

推断 concept_name 的三个来源:
1. **变量名 / 注释**: `prefix` 是 "context"，`suffix` 是 "queried from noise"
2. **论文章节标题**: 论文里"Architecture"/"Loss"那两节常常有官方命名
3. **README 的 overview 图**: 多数 repo 都有，照搬

参考 `references/topology-inference.md` Step 3-4。

## Step 4 — 写 tooltip + what + blurb

每个 leaf 的三层文字（按 [explanation-style.md](explanation-style.md) 约定）:

- **`tooltip`** (~15 字): 鼠标 hover 一行可见。例: `"投影 noisy action 每帧到 expert hidden=720"`
- **`what`** (~80-150 字): 抽屉里的"它在做什么"段。**中文要源自论文 / 注释 / 用户原话**，不要 LLM 凭空想象（HANDOFF 用户偏好）
- **`blurb`** (composite 才有，~50-100 字): 这个 composite 的整体作用 + 几个子节点的关系

## Step 5 — 关键 leaf 补 formula / callout

只给**真正有数学**的 leaf 加 `formula` 字段（LaTeX，KaTeX 渲染）:

```python
"formula": r"\text{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}}\right)V",
```

`callout` 是抽屉里的"**这个模块为什么重要 / 别人怎么没做对**"段落：

```python
"callout": "<strong>这是整个 SmolVLA 的灵魂模块。</strong>VLM 的预训练知识就是通过这里被一层一层「灌」进 action expert 的...",
```

## Step 6 — 标 is_heart + training_only

`is_heart`: 整张图最多 1 个，是**模型核心计算单元**（占大半参数 + forward 中的 merge point）。SmolVLA 是 `vlm_with_expert`，CLIP 是 image+text encoder 并列（没有单一心脏）。

`training_only`: 只在训练态存在的步骤（loss / backward / 噪声采样）。给虚线 + 灰底。

加在对应 stage 字典里:
```python
{"nodes": ["vlm_expert"], "is_heart": True},
{"nodes": ["flow_loss"], "training_only": True},
```

## Step 7 — 写 story_line（顶部 blurb）

一句话开场，让读者瞬间抓 frame。模板:

> **「模型」不直接预测「目标」，而是「关键 trick」——「训练时怎么做」。**

例:
```python
"story_line": (
    'SmolVLA <strong>不直接预测动作</strong>, 而是预测从噪声到真动作的'
    '<strong>速度场</strong> \\(v_t\\) —— 训练时随机采时间 \\(t\\), '
    '把真实动作 \\(a\\) 和噪声 \\(\\varepsilon\\) 按 \\(t\\) 混合得到 \\(x_t\\), '
    '让模型还原这个混合背后的速度方向 \\(u_t = \\varepsilon - a\\)。'
),
```

来源:
1. 论文 abstract 末尾两句
2. README 开头段落
3. 都没有 → 读 loss 函数 + forward 末尾，反推 training objective，套模板

## Step 8 — 验证清单（提交前过一遍）

- [ ] 跑 `python scripts/render_html.py <trace.json>`，开 HTML 看 Y-shape 拓扑是不是对的（不是 6 个串行 box）
- [ ] 每个方块 5 秒能看懂作用？concept_name 表达"做什么"不是"叫什么"
- [ ] 维度名是语义的（`[batch=16, seq=177]` 不是 `[16, 177]`）
- [ ] 心脏方块给大字号了（视觉上一眼就找到）
- [ ] 训练-only 方块虚线 + 低饱和度
- [ ] story_line 一句话能 frame 整个 pipeline
- [ ] 选中页面文字 → 弹 FAB → 问 AI，AI 收到的 context 跟选中文字对得上

---

## 输出格式

把改好的 `tree_skeleton.py` 重命名为 `render/tree_<repo>.py`（例: `tree_clip.py`），然后在 `render/__init__.py` 改 import:

```python
# from .tree_smolvla import TREE
from .tree_<repo> import TREE
```

或者搞两套并存，看哪条 trace 调哪个 TREE。

完成后跑 `python scripts/render_html.py outputs/traces/<repo>/trace_<ts>.json`，输出的 HTML 直接打开看效果，对照 checklist 一项项过。
