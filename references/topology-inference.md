# Topology inference — 怎么从论文 + 源码 + trace 推出"正确的"顶层架构图

> 这份文档教本地 Claude 在任意 PyTorch repo 上产出一张让人**第一眼就懂**的顶层架构图。核心命题是：trace JSON 给的是"代码模块的调用顺序"，但用户需要的是"数据怎么变成 loss"。两者经常不一样，这里讲怎么从原材推到合适。

---

## 输入

1. **trace JSON**（`module_tree` + `module_events` + `first_batch`）→ 工程真相
2. **论文 / README / docstring**（如果有）→ 概念真相
3. **入口文件源码**（训练脚本 + 顶层 model class 的 `forward`）→ 执行顺序

三者缺一不可。只用 trace JSON 推不出语义；只读论文给不出代码真实跑的样子。

---

## 输出

一张 SVG 顶层图 + 一份元数据 JSON，分别：

- `story_line`：1-2 句话开场白（顶部 blurb）
- `nodes`：每个顶层方块的 `concept_name` / `code_name` / `purpose_line` / `is_composite` / `is_training_only`
- `edges`：连接 + 张量 shape 标注（语义化维度名）
- `topology`：linear / Y-shape / fork-join / multi-branch
- `emphasis`：哪个方块是"心脏"，给大给粗

---

## Step 1 — 读 forward 的"语义顺序"，而不是 trace 顺序

**反模式**：trace JSON 按 module 的 `forward_pre` hook 触发时间排序。子模块会比父模块**早**进入但**晚**返回。如果直接按事件顺序画拓扑图，会得到一堆"叶子叠在父节点"的歪布局。

**正确做法**：读顶层 model class 的 `forward` 方法源码，按**语法顺序**抽取顶层操作。这才是作者**想要的顺序**。

例子（SmolVLA 的 `VLAFlowMatching.forward`）：
```python
def forward(self, batch):
    images, state, action, lang = self.prepare_inputs(batch)
    prefix_tokens = self.embed_prefix(images, lang, state)
    noise, t = self.sample_noise_and_time(batch_size)
    x_t = (1 - t) * noise + t * action
    suffix_tokens = self.embed_suffix(x_t, t)
    h = self.vlm_with_expert(prefix_tokens, suffix_tokens)
    v_t = self.action_out_proj(h)
    loss = F.mse_loss(v_t, action - noise)
    return loss
```

直接把这 6 个调用点产出 6 个顶层方块。每一行的语义来自变量名和源码上下文，而不是 trace 里的类名。

## Step 2 — 识别拓扑形状（Y / fork-join / linear）

不是所有数据流都是直线。**正确做法是从 forward 源码反推**（这一步唯一可靠）—— 不要试图把模型归类到某个预设模板，那是锚定陷阱。

下面的拓扑词汇表只是**统一表达**用的（让你描述拓扑时有名字可用），**不是 checklist 让你逐个套**：

### 拓扑词汇表（reference only）

```
Linear:                A → B → C → D
                       每一步唯一消费上一步, 无分叉, 无合并

Y-shape (merge):       A           D
                          ╲       ╱
                           B  →  C
                          ╱       ╲
                       E           F
                       两个独立来源, 在某一步被合并消费
                       例: VLA (observation + action)、joint encoder

Fork-join:                   ╭─ B ─╮
                       A ───┤      ├─ D
                            ╰─ C ─╯
                       一个变量被复制走多条路再合并
                       例: ResNet residual、Inception module

Dual-encoder:          A → B ──╮
                                ├─ Loss(B_out, F_out)
                       E → F ──╯
                       两路平行处理, 不合并, 在 loss 处比较
                       例: CLIP、Siamese network

Encoder-decoder:       A → B → C ───╮
                                     ├─ G → H → I
                       D → E → F ───╯
                       encoder 输出做 cross-attention 给 decoder
                       例: T5、原版 Transformer、Whisper

U-Net (skip):          A ─┬──────────────┬─ G
                          │              │
                          B ─┬────────┬─ F
                              │      │
                              C ─── E
                                D
                       深层与浅层 skip 拼接, 边降采样边保细节
                       例: U-Net、stable diffusion 的 UNet

Mixture-of-Experts:    A ─→ Router ─→ Expert_1, ..., Expert_k ─→ combine
                       一个 router 决定哪些 expert 处理样本, 输出加权
                       例: Mixtral 8x7B、Switch Transformer、Gshard

Multi-task heads:           ╭─ Head_A (cls)
                       A → B ─├─ Head_B (det)
                            ╰─ Head_C (seg)
                       共享 backbone 给多个独立 head
                       例: Mask R-CNN、HRNet
```

罕见架构（mamba / RWKV / 状态空间 / 自定义混合）很可能不属于上面任何一类 —— 那就**老老实实画出它独特的 forward 流**，不要硬塞模板。

### 检测方法（唯一可靠路径）

读 `<root>` class 的 `forward()` 源码, 把每条 `=` 左边的变量当作图节点, 右边引用的变量当作入边, 画一张**变量依赖图**。然后:

- 一个节点的入边来自**两个互不依赖的源头** → 那就是 merge / Y-shape
- 一个变量被同时传给**多个调用** → fork
- 两条链直到 loss 才碰面 → dual-encoder
- 早期变量被**远后**的调用消费 → skip connection

**反模式（v3 SmolVLA 错过的）**：把 Y 形压平成 `prepare → embed_prefix → embed_suffix → vlm_expert` 的线性链。`embed_suffix` 不消费 `embed_prefix` 的输出, 画成串行是错的。

## Step 3 — 给每个方块起两个名字

每个方块都需要两个 label：

- **`concept_name`**：用户**第一眼**要看的，揭示"做什么"
- **`code_name`**：用户**点击后**要看的，揭示"代码里叫什么"

| Code 里叫什么 | 当 concept_name | 当 code_name |
|---|---|---|
| `embed_prefix` | 观测编码 / Encode observation | `embed_prefix` |
| `embed_suffix` | 动作加噪 / Noise + action mix | `embed_suffix` |
| `vlm_with_expert` | VLM + action expert | `VLAFlowMatching.vlm_with_expert` |
| `action_out_proj` | 预测速度场 / Predict velocity | `Linear(960, 6)` |
| `F.mse_loss(v, a-e)` | Flow loss | `F.mse_loss` |

**推断 concept_name 的方法**：让 LLM 读三个来源——
1. **变量名 + 注释**（lerobot 里 `prefix` 显然是"context"，`suffix` 显然是"queried-from-noise"）
2. **论文章节标题**（SmolVLA 的论文有 "Architecture" 和 "Flow matching loss" 两节）
3. **README 的 overview 图**（很多 repo 都有，是作者自己给出的概念名）

如果都没有，至少**反映作用而不是变量名**：`prepare_state` -> "归一化状态"（`norm_state_with_dataset_stats` -> "归一化状态"，不是"用数据集统计归一化状态"，太长）。

## Step 4 — 写 purpose_line（hover 一下直可见的那一行）

每个方块旁边写一行"做什么"。规则：

- **<= 5 个词**（中文 <= 10 字）
- **动词领头**：`vision + language + state` 不如 `编码 vision/lang/state`，前者只是名词列表，没说在干啥
- **不重复 concept_name**：concept_name 是"观测编码"，purpose_line 写"vision + language + state"补充信息，不要重复 "encode" 这个动词

**反模式**：把方块上下都填满。每个方块 2-3 个信息单元就够了——concept_name + code_name + purpose_line（在 hover/抽屉）。

## Step 5 — 标注张量 shape（语义化维度）

每条连线上标一组 shape，规则：

1. **维度名优先**：`[batch=16, seq=177, hidden=960]`，不是 `[16, 177, 960]`
2. **维度名从源码推**：
   - `x.shape[0]` 在 batch loop 里 -> `batch`
   - 4D 张量含 `h`、`w` 命名 -> `[batch, channel, height, width]`
   - 序列模型里的第二维 -> `seq` 或 `tokens`
   - `action` 上下文 -> `chunk` / `action_dim`
3. **每个 shape 都从 trace 拿真实捕获**，不要凭想象。trace JSON 的 `inputs[*].shape` 是 ground truth
4. **shape 跨节点不变就不重复标**。只在 shape 真的变了的地方标注（reshape / projection / merge）

## Step 6 — 判定 composite vs leaf

**硬规则**：trace 的 `module_tree` 里有子节点 -> composite（可下钻）；没子节点 -> leaf（直接看详情）。

**软覆盖**（用一份配置 list）：
- `nn.MultiheadAttention` -> 强制 leaf（内部就几个 Linear，下钻没意义）
- 任何 functional 调用（`F.scaled_dot_product_attention`、`F.mse_loss`）-> leaf，加 `functional` badge
- 自实现的 attention（包含 `q_proj` / `k_proj` 等子模块）-> 保持 composite（值得下钻看 joint attention 细节）

## Step 7 — 标出"心脏方块"和"训练特有方块"

**心脏方块**：整个模型的核心计算单元。视觉上**给大字号**（更高的 h，更粗的 stroke），用户第一眼看到它。判定方法：

- 占大半参数（从 trace 的 module params 推）
- forward 中被传入了所有的输入流（merge point）
- 论文里架构图最显眼的那个 box

例子：SmolVLA 是 `vlm_with_expert`，Llama 是单个 decoder layer，CLIP 训练里没有单一心脏（image + text encoder 都重要）。

**训练特有方块**：只在训练时存在的步骤，给虚线 + 低饱和度。判定方法：
- 在 `if self.training:` 分支里
- 涉及 `loss` 函数
- 涉及 `backward()` / `optimizer.step()`
- 噪声采样（虽然在 forward 里但只在 flow matching/diffusion 训练时使用）

让用户一眼看出"主干 = X 个实线方块，Y 个虚线是训练才有"。

## Step 8 — 写 story_line（顶部 blurb）

这页面第一句话。**一句话**，让读者瞬间抓 frame。模板：

> **「模型」不直接预测「目标」，而是「关键 trick」——「训练时怎么做」。**

例子：

- SmolVLA: "SmolVLA 不直接预测动作，而是预测从噪声到真动作的速度场——训练时随机采时间 t、把动作和噪声按 t 混合、让模型还原这个混合背后的速度。"
- DDPM: "DDPM 不直接生成图像，而是学习从噪声一步步去噪——训练时给图像随机加 t 步噪声，让模型预测加的噪声是什么。"
- CLIP: "CLIP 不学单向映射，而是学 image 和 text 在同一向量空间里对齐——训练时让一对正的匹配 (image, text) 的 embedding 接近，其他都对远离。"

**story_line 从哪来**：
1. 论文 abstract 的最后两句往往就是
2. README 的开头段落
3. 如果都没有，让 LLM 看 loss 函数 + forward 末尾，反推训练 objective，再用上面模板套句

---

## 实现：一份可重用的 prompt 模板

把 Step 1-8 编码成一个 prompt，发给 LLM 一次性产出元数据 JSON。模板放 `references/topology-prompt.md`，本地 Claude 每次跟新 repo 都调它。

prompt 大致结构：

```
你正在为一个 PyTorch repo 生成顶层架构图的元数据。

输入：
1. 顶层 forward 方法源码（粘下面）
2. trace JSON 的 module_tree 摘要（粘下面）
3. README / 论文 abstract（粘下面，可选）

任务：
对每个 forward 中的顶层操作，输出一个 node 对象，字段：
- concept_name (<= 10 字)
- code_name
- purpose_line (<= 5 词)
- is_composite (bool)
- is_training_only (bool)
- is_heart (bool, 整张图最多 1 个)
- input_shape_label / output_shape_label（语义化维度名）

按 forward 的语法顺序排序。

识别拓扑：观察哪些 node 的输入来自不同源头，标出 Y 形 / fork-join。

最后输出 story_line：一句话开场，遵循模板"「模型」不直接...而是..."

返回纯 JSON，不要解释。
```

跟着这份元数据，渲染器直接给 SVG，不需要再用 LLM 做布局。

---

## 验证清单

产出图前过这遍：

- [ ] Y 形 / 分叉：如果有并行分支，是否真的画成了并列方块？
- [ ] concept_name：从图上能否 5 秒看懂模型作用？
- [ ] purpose_line：每个方块都有？且不重复 concept_name？
- [ ] shape 标注：维度名都是语义化的，不是裸数字？
- [ ] 心脏方块：找到了吗？给大了吗？
- [ ] 训练-only：虚线 + 低饱和度？
- [ ] story_line：一句话能 frame 整个 pipeline？

如果有任何一项不满足，回到对应 step 修。
