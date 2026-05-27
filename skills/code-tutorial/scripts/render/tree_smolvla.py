"""Hand-designed conceptual tree for SmolVLA.

This is the SmolVLA-specific topology data (~68 nodes describing the Y-shape
pipeline from raw_batch through embed_prefix/embed_suffix → vlm_expert →
action_proj → flow_loss → backward_step).

Real shape / src / callCount are filled in by enrich_with_trace().

NOTE (P1 work): Generalising this skill to other PyTorch repos means writing
a TREE dict like this one for that repo; see references/topology-inference.md
for the Step 1-8 process. infer_tree.py is the planned tool to make this
semi-automatic.
"""

from __future__ import annotations

from typing import Any


TREE: dict[str, dict[str, Any]] = {
    "pipeline": {
        "type": "composite",
        "name": "SmolVLA 一次训练 step",
        "sub": "VLAFlowMatching · forward + backward",
        "tooltip": "",
        "blurb": "",
        "layout": "stages",
        "story_line": (
            'SmolVLA <strong>不直接预测动作</strong>, 而是预测从噪声到真动作的'
            '<strong>速度场</strong> \\(v_t\\) —— 训练时随机采时间 \\(t\\), '
            '把真实动作 \\(a\\) 和噪声 \\(\\varepsilon\\) 按 \\(t\\) 混合得到 \\(x_t\\), '
            '让模型还原这个混合背后的速度方向 \\(u_t = \\varepsilon - a\\)。'
        ),
        "stages": [
            {"nodes": ["raw_batch"]},
            {"nodes": ["embed_prefix", "embed_suffix"],
             "outgoing_labels": {
                 "embed_prefix": "prefix [batch=16, seq=177, hidden=960]",
                 "embed_suffix": "suffix [batch=16, chunk=50, hidden=720]",
             }},
            {"nodes": ["vlm_expert"], "is_heart": True,
             "outgoing_labels": {
                 "vlm_expert": "suffix_out [batch=16, chunk=50, hidden=720]",
             }},
            {"nodes": ["action_proj"],
             "outgoing_labels": {
                 "action_proj": "v_t [batch=16, chunk=50, action_dim=32]",
             }},
            {"nodes": ["flow_loss"], "training_only": True,
             "outgoing_labels": {
                 "flow_loss": "scalar loss",
             }},
            {"nodes": ["backward_step"], "training_only": True},
        ],
        "children": [
            "raw_batch", "embed_prefix", "embed_suffix",
            "vlm_expert", "action_proj", "flow_loss", "backward_step",
        ],
    },

    "raw_batch": {
        "type": "leaf",
        "name": "Raw batch",
        "sub": "dict from DataLoader",
        "tooltip": "DataLoader 出来的原始字段：两路图像、本体 state、action chunk、语言 prompt token。",
        "what": "训练循环每步从 LeRobotDataset 抽一个 batch。SO101 这条数据集每条 sample 含两路相机（俯视 fixed + 末端 wrist）、一帧 6 维本体 state、未来 50 帧 6 维 action chunk，加任务文本 prompt 已 tokenize 好（48 token 上限）。",
        "shape_html": (
            '<span class="dim">fixed</span> [batch=16, c=3, h=640, w=480] uint8<br>'
            '<span class="dim">wrist</span> [batch=16, c=3, h=480, w=640] uint8<br>'
            '<span class="dim">state</span> [batch=16, t=1, dim=6] float32<br>'
            '<span class="dim">action</span> [batch=16, chunk=50, action_dim=6] float32<br>'
            '<span class="dim">lang_tokens</span> [batch=16, seq=48] int64'
        ),
        "qualname": "lerobot.datasets.LeRobotDataset → dict (collate_fn)",
        "src": "lerobot/datasets/lerobot_dataset.py",
        "callCount": 1,
    },

    "prepare_images": {
        "type": "leaf",
        "name": "Resize + normalize",
        "sub": "resize_with_pad · [-1, 1]",
        "tooltip": "对每路相机：保比缩放 + 0 填到 512×512, 再把 [0,1] 映到 [-1,1]（SigLIP 期望）。",
        "what": "fixed 是竖屏 640×480, wrist 横屏 480×640, 两路 shape 不同, 用 resize_with_pad 统一到 512×512（短边 0 填）。然后值域从 [0, 1] 线性映到 [-1, 1] —— SmolVLM 视觉塔（基于 SigLIP）训练时输入就是这个区间。每路还有一个 (batch,) 的 img_mask 标识图像存在与否。",
        "qualname": "SmolVLAPolicy.prepare_images",
        "src": "lerobot/policies/smolvla/modeling_smolvla.py:470",
        "callCount": 1,
        "shape_html": (
            'Per camera: [batch=16, c=3, h=640/480, w=480/640] uint8 '
            '<span class="arrow">→</span> [batch=16, c=3, h=512, w=512] float32 ∈ [−1,1]'
        ),
    },

    "prepare_state": {
        "type": "leaf",
        "name": "Pad state",
        "sub": "6 → 32",
        "tooltip": "取最后一帧 + 尾部补 0 到 max_state_dim=32（state_dropout 默认关）。",
        "what": "SO101 的本体 state 是 6 维（5 关节 deg + gripper 行程 %）。SmolVLA 内部按 32 维处理跨机型, 所以 6 维尾部补 0。训练态如果开了 state_dropout 还会以概率 p 把整条 state 置零, 强迫模型从视觉 + 语言推动作 —— v18 没开。",
        "qualname": "SmolVLAPolicy.prepare_state",
        "src": "lerobot/policies/smolvla/modeling_smolvla.py:539",
        "callCount": 1,
        "shape_html": (
            '[batch=16, t=1, dim=6] '
            '<span class="arrow">→</span> [batch=16, state_dim=32]'
        ),
    },

    "prepare_action": {
        "type": "leaf",
        "name": "Pad action",
        "sub": "6 → 32",
        "tooltip": "Action chunk 最后一维 pad 到 max_action_dim=32；chunk=50 不变。",
        "what": "Action chunk 也走 32 维统一接口, 6 维尾部补 0 到 32。chunk_size=50 是 SmolVLA / π0 系的标配 —— 一次预测未来 50 帧动作, 大约 1-2 秒的轨迹片段。后面 flow matching 全程都在这个 [batch, 50, 32] 张量上算速度场。",
        "qualname": "SmolVLAPolicy.prepare_action",
        "src": "lerobot/policies/smolvla/modeling_smolvla.py:551",
        "callCount": 1,
        "shape_html": (
            '[batch=16, chunk=50, action_dim=6] '
            '<span class="arrow">→</span> [batch=16, chunk=50, action_dim=32]'
        ),
    },

    "embed_prefix": {
        "type": "composite",
        "name": "观测编码",
        "sub": "embed_prefix · vision + lang + state",
        "tooltip": "把两路相机、语言指令、本体 state 编成 prefix tokens 序列, 喂给冻结的 VLM 主干。",
        "blurb": "Prefix 是模型「能看到」的那段 token 序列。两路相机各跑一遍 SmolVLM 视觉塔（12 层 ViT, 1024 patch token → connector pixel shuffle 压到 64 token）, 加 48 个语言 token 和 1 个 state token, 合 prefix seq=177。视觉塔参数冻结, prefix 这条链路微调时只更 state_proj 那一个 Linear。",
        "children": ["prepare_images", "vision_tower", "connector", "lang_embed", "prepare_state", "state_proj"],
    },

    "vision_tower": {
        "type": "composite",
        "name": "Vision tower",
        "sub": "SmolVLM2 ViT · 12 layers",
        "tooltip": "SmolVLM 的视觉编码器, patch_embed → 12 层 self-attention → post LayerNorm。每路相机各跑一次, 微调时冻结。",
        "blurb": "SmolVLM2 自带的视觉 backbone, 也就是 SigLIP 派。先 16×16 stride conv 把 512² 切成 32×32 patch（共 1024 token, hidden=768）, 再过 12 层标准 ViT block（pre-LN + multi-head self-attn + MLP）。微调全程冻结, 只前向抽特征。trace 里两路相机各跑一遍 → 这棵子树被调 2 次。",
        "children": ["patch_embed", "vision_layers", "vision_post_layernorm"],
    },

    "patch_embed": {
        "type": "leaf",
        "name": "Patch embed",
        "sub": "Conv2d 16×16 stride 16",
        "tooltip": "16×16 stride=16 的 Conv2d, 把 512×512 RGB 切成 32×32 个 patch token, 每个 768 维。",
        "what": "做 ViT 经典的 patch 切分。stride=kernel=16 的 Conv2d 等价于「把图切成 32×32 网格, 每格 16×16 像素拼平后过 Linear 投影到 768」。从 trace 看 Conv 出来形状是 (16, 768, 32, 32), 后面会 flatten 成 (16, 1024, 768) 的 token 序列。",
        "trace_name": "model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding",
        "in_hint": "image",
        "out_hint": "vision_tokens",
    },

    "vision_layers": {
        "type": "composite",
        "name": "ViT encoder layer",
        "sub": "pre-LN + SA + MLP",
        "tooltip": "标准 ViT block: LayerNorm → self-attention → residual → LayerNorm → MLP → residual。",
        "blurb": "ViT transformer 主体。12 层结构完全相同, trace 抓到 12 个独立实例, 这里展平只看「一层是什么样的」。注意 vision tower 是标准 multi-head attention（不是 GQA）, 12 个 head 共享 K/V。",
        "children": ["vis_norm1", "vis_self_attn", "vis_norm2", "vis_mlp"],
        "repeat": 12,
    },

    "vis_norm1": {
        "type": "leaf",
        "name": "LayerNorm",
        "sub": "pre-attention",
        "tooltip": "attention 前的 LayerNorm, 标准 ViT 用 LN 不是 RMSNorm。",
        "what": "Vision tower 的 pre-attn LayerNorm。<strong>对每个 token 独立操作</strong>: 沿 hidden 维度算均值 \\(\\mu\\) 和方差 \\(\\sigma^2\\), 减去均值除以标准差归到零均值单位方差, 然后过一个可学的仿射 \\(\\gamma \\odot \\hat{x} + \\beta\\)。这样 attention 输入永远在一个 well-defined 数值范围, 是 transformer 训练稳定的关键。LN 跟 BatchNorm 区别: LN 沿 feature 维度算（每个 token 独立）, BN 沿 batch 维度算（每个 feature 独立）—— transformer 用 LN 是因为 batch 维度不一定可靠（变长序列、推理时 batch=1）。",
        "trace_name": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.layer_norm1",
        "in_hint": "vision_tokens",
        "out_hint": "vision_tokens",
        "formula": r"\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \quad y_i = \gamma_i \hat{x}_i + \beta_i",
    },

    "vis_self_attn": {
        "type": "composite",
        "name": "Self-attention",
        "sub": "standard MHA · 12 heads",
        "tooltip": "标准多头自注意力, Q/K/V 三个 Linear 各占 hidden, SDPA 做 attention, out_proj 收尾。",
        "blurb": "Vision tower 的 self-attention 块。所有 1024 个 patch token 互相 attend, 模型借此聚合空间信息。Q/K/V 三路 Linear 都从 768 投到 768, 然后 reshape 成 12 head × 64 dim, F.scaled_dot_product_attention 算注意力, out_proj 投回 768。",
        "children": ["vis_q", "vis_k", "vis_v", "vis_sdpa", "vis_out"],
    },

    "vis_q": {
        "type": "leaf", "name": "Q proj", "sub": "Linear 768→768",
        "tooltip": "Q 投影, 12 head × 64 dim.",
        "what": "把每个 patch token 投影成 query 向量, 后续会 reshape 成 12 个 head 各 64 维。",
        "trace_name": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.q_proj",
        "in_hint": "vision_tokens", "out_hint": "vision_tokens",
    },
    "vis_k": {
        "type": "leaf", "name": "K proj", "sub": "Linear 768→768",
        "tooltip": "K 投影, 跟 Q 同结构。",
        "what": "投影出 key 向量。Vision tower 是标准 MHA, 每个 head 都有独立的 K（没用 GQA）。",
        "trace_name": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.k_proj",
        "in_hint": "vision_tokens", "out_hint": "vision_tokens",
    },
    "vis_v": {
        "type": "leaf", "name": "V proj", "sub": "Linear 768→768",
        "tooltip": "V 投影。",
        "what": "投影出 value 向量, 跟 K 一样 12 head 各 64 维。",
        "trace_name": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.v_proj",
        "in_hint": "vision_tokens", "out_hint": "vision_tokens",
    },
    "vis_sdpa": {
        "type": "leaf", "name": "SDPA", "sub": "F.scaled_dot_product_attention",
        "tooltip": "PyTorch 内置的快速 attention 实现, 自动挑 FlashAttention / mem-efficient。",
        "functional": True,
        "what": "attention 的核心计算。每个 query 跟所有 key 点积, 缩放 / softmax / 加权 value。trace 顶部统计的 SDPA op 主要来自这里（12 层 × 2 路相机 = 24 次）。",
        "formula": r"\text{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}}\right)V",
        "callout": "PyTorch 内置的 F.scaled_dot_product_attention 不是 nn.Module, 用普通 module hook 抓不到。这次 trace 通过 monkey-patch 把它单独 wrap 了一遍才能记录。",
        "src": "transformers/integrations/sdpa_attention.py:92",
        "callCount": 24,
        "shape_html": (
            'Q, K, V: [batch=16, heads=12, tokens=1024, head_dim=64] '
            '<span class="arrow">→</span> '
            '[batch=16, heads=12, tokens=1024, head_dim=64]'
        ),
    },
    "vis_out": {
        "type": "leaf", "name": "Output proj", "sub": "Linear 768→768",
        "tooltip": "Attention 输出投影, 把多头结果合回 hidden 维。",
        "what": "把 12 个 head 的输出 concat 起来再过一个 Linear 合回 768 维, 是 attention 块的最后一步。",
        "trace_name": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.out_proj",
        "in_hint": "vision_tokens", "out_hint": "vision_tokens",
    },

    "vis_norm2": {
        "type": "leaf", "name": "LayerNorm", "sub": "pre-MLP",
        "tooltip": "MLP 前的第二个 LayerNorm。",
        "what": "MLP 前的 pre-LN, 跟 pre-attn LN 结构相同, 权重独立。Vision tower 用 LN 不是 RMSNorm。",
        "trace_name": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.layer_norm2",
        "in_hint": "vision_tokens", "out_hint": "vision_tokens",
    },

    "vis_mlp": {
        "type": "composite", "name": "MLP", "sub": "GELU FFN",
        "tooltip": "标准 ViT FFN: fc1 升到 4×, GELU, fc2 降回 hidden。",
        "blurb": "ViT 的 feedforward 块, 标准两层 MLP 中间夹 GELU。fc1 把 hidden 从 768 升到 mlp_dim=3072（论文里通常 4×, 跟实际 trace 一致）, GELU 激活, fc2 降回 768。",
        "children": ["vis_fc1", "vis_gelu", "vis_fc2"],
    },
    "vis_fc1": {
        "type": "leaf", "name": "FC1", "sub": "Linear 768→3072",
        "tooltip": "MLP 升维投影。",
        "what": "把 hidden 768 升到中间维 3072（约 4×, 论文 FFN 标准放大比）。展宽是为了让非线性有更多容量学复杂特征。",
        "trace_name": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.fc1",
        "in_hint": "vision_tokens", "out_hint": "vision_tokens",
    },
    "vis_gelu": {
        "type": "leaf", "name": "GELU", "sub": "activation (tanh approx)",
        "tooltip": "非线性激活, ViT 标配。",
        "what": "GELU activation, \\(\\mathrm{GELU}(x) = x \\cdot \\Phi(x)\\), 比 ReLU 平滑、在小负值区间也有梯度。Vision tower 用 tanh 近似版本（GELUTanh）。无可训参数, 不在 trace 里出现独立 forward call。",
        "trace_name": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.activation_fn",
        "in_hint": "vision_tokens", "out_hint": "vision_tokens",
        "act_chart": "gelu",
        "formula": r"\mathrm{GELU}(x) = \tfrac{1}{2} x \left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}} \, (x + 0.044715\, x^3)\right)\right)",
    },
    "vis_fc2": {
        "type": "leaf", "name": "FC2", "sub": "Linear 3072→768",
        "tooltip": "MLP 降维, 收回 hidden。",
        "what": "降维 Linear, 把 MLP 中间维 3072 投回 hidden 768, 然后加 residual 接下一层。",
        "trace_name": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.fc2",
        "in_hint": "vision_tokens", "out_hint": "vision_tokens",
    },

    "vision_post_layernorm": {
        "type": "leaf", "name": "Post-LN", "sub": "final LayerNorm",
        "tooltip": "12 层结束后的最后一道 LayerNorm。",
        "what": "ViT 在 encoder 最后一层 transformer 之后, 还有一道 post-LN, 把整体输出再标准化一遍。这是 SmolVLM 用的「pre-LN + final post-LN」混合方案。",
        "trace_name": "model.vlm_with_expert.vlm.model.vision_model.post_layernorm",
        "in_hint": "vision_tokens", "out_hint": "vision_tokens",
    },

    "connector": {
        "type": "leaf",
        "name": "Connector",
        "sub": "pixel shuffle + Linear 768→960",
        "tooltip": "把 1024 patch token 压成 64 token（pixel shuffle), 顺手升 hidden 到 VLM body 的 960。",
        "what": "视觉塔吐出来 1024 个 token 太多, connector 用 pixel shuffle 在 4×4 空间块上合并（1024/16 = 64）, 信息没丢只是重排。同时 hidden 从 768 调整到 VLM body 的 960。这一步是 SmolVLM 把视觉塞进 LLM 上下文不爆的关键。",
        "trace_name": "model.vlm_with_expert.vlm.model.connector",
        "in_hint": "connector_in", "out_hint": "connector_out",
        "callout": "<strong>Pixel shuffle 的妙处：</strong>不是 pool 平均、不是 attention 聚合, 而是 reshape <code>(B, H, W, D) → (B, H/4, W/4, 16·D)</code>, 再 Linear 投到目标 hidden。空间信息打散进通道, 一字节都没扔。",
    },

    "lang_embed": {
        "type": "leaf",
        "name": "Lang embed",
        "sub": "embed_tokens · 49152→960",
        "tooltip": "把语言 token id 查 embedding 表得到 960 维向量。",
        "what": "任务文本（例如 \"Pick up the cube...\"）已经在 dataset processor 里 tokenize 成 id 序列, 这里只做 lookup。embedding table 是 VLM 共享的, 词表 ~49k, 输出维度 960 跟 VLM body 对齐。",
        "trace_name": "model.vlm_with_expert.vlm.model.text_model.embed_tokens",
        "in_hint": "lang_tokens", "out_hint": "prefix",
    },

    "state_proj": {
        "type": "leaf",
        "name": "State proj",
        "sub": "Linear 32→960",
        "tooltip": "本体 state 投影成 1 个 prefix token（hidden=960）。",
        "what": "把 32 维 state 用一个 Linear 投到 hidden 960 的单个 token, 挂在 prefix 序列末尾。state_proj 是微调里 prefix 链路上**唯一**会被更新的可训参数 —— 视觉塔冻结、lang embedding 冻结、connector 跟在 VLM 里也冻结, 只有这里和 expert 那一侧的 expert 自己有梯度。",
        "trace_name": "model.state_proj",
        "in_hint": "state_padded", "out_hint": "state_proj_out",
    },

    "embed_suffix": {
        "type": "composite",
        "name": "动作加噪",
        "sub": "embed_suffix · noisy action + time",
        "tooltip": "把真实动作 a 跟噪声 ε 按时间 t 混合得到 \\(x_t\\), 编成 suffix tokens 喂给 action expert。",
        "blurb": "Suffix 是「模型要预测」那段的输入 token。先 sample noise \\(\\varepsilon\\) 和时间 \\(t\\), 算出插值 \\(x_t = t\\varepsilon + (1-t)a\\); 然后 action_in_proj 把 \\(x_t\\) 每帧投到 expert hidden=720, 时间 \\(t\\) 用正弦位置编码也变 720 维, 两者 concat 后过一个小 MLP 融合, 得到 50 个 suffix token。",
        "children": ["prepare_action", "sample_noise", "sample_time", "interpolate_xt", "action_in_proj", "sinusoidal_time", "action_time_mlp"],
    },

    "sample_noise": {
        "type": "leaf",
        "name": "Sample ε",
        "sub": "N(0, I)",
        "tooltip": "高斯噪声, shape 跟 actions 一致 (16, 50, 32)。",
        "what": "对每条 sample 独立抽一份 ε ~ N(0, I), shape 跟 action chunk 完全一样。这是 flow matching 的「源端分布」, 训练时不同的 (a, ε) 组合让模型在整个噪声→动作概率管道上覆盖速度场。",
        "qualname": "VLAFlowMatching.sample_noise",
        "src": "lerobot/policies/smolvla/modeling_smolvla.py:691",
        "callCount": 1,
        "formula": r"\varepsilon \sim \mathcal{N}(0, I), \quad \varepsilon \in \mathbb{R}^{16 \times 50 \times 32}",
        "shape_html": '<span class="arrow">→</span> [batch=16, chunk=50, action_dim=32] float32',
    },

    "sample_time": {
        "type": "leaf",
        "name": "Sample t",
        "sub": "Beta(1.5, 1.0)·0.999 + 0.001",
        "tooltip": "扩散时间 t ~ Beta(1.5, 1.0), 轻度偏向 t→1 那一侧, shape (batch,).",
        "what": "每条 sample 抽一个独立的 t ∈ (0.001, 1.0)。用 Beta(1.5, 1.0) 而不是均匀分布, 是想让训练多见到「靠近真实动作端」的插值点 —— 经验上这样训出来的 chunk 在 t 小的时候没那么抖。",
        "qualname": "VLAFlowMatching.sample_time",
        "src": "lerobot/policies/smolvla/modeling_smolvla.py:701",
        "callCount": 1,
        "formula": r"t \sim 0.999 \cdot \mathrm{Beta}(1.5, 1.0) + 0.001",
        "shape_html": '<span class="arrow">→</span> [batch=16] float32',
    },

    "interpolate_xt": {
        "type": "leaf",
        "name": "x_t & u_t",
        "sub": "linear interpolation",
        "tooltip": "x_t = t·ε + (1−t)·a (输入), u_t = ε − a (目标速度)。两者 shape 跟 actions 一致。",
        "what": "flow matching 的数学锚点。x_t 是真实 action a 朝纯噪声 ε 走了 t 比例那么远的中间状态; u_t 是从 a 指向 ε 的方向向量 —— 这就是模型要学的「速度场」target。注意 u_t 跟 t 无关, 是个常向量。",
        "qualname": "VLAFlowMatching.forward (inline)",
        "src": "lerobot/policies/smolvla/modeling_smolvla.py:865",
        "callCount": 1,
        "formula": r"x_t = t\,\varepsilon + (1-t)\,a, \qquad u_t = \varepsilon - a",
        "callout": "<strong>linear conditional flow matching</strong> 的数学很干净 —— 选直线路径 ε ↔ a, target velocity 就是 ε − a 这种 closed-form 常向量。比 score matching / diffusion 省一个 forward-sim 训练成本。",
        "shape_html": 'x_t, u_t: [batch=16, chunk=50, action_dim=32] float32',
    },

    "action_in_proj": {
        "type": "leaf",
        "name": "Action in proj",
        "sub": "Linear 32→720",
        "tooltip": "把 noisy action 每帧投到 expert hidden=720。",
        "what": "Linear 把 action chunk 的每帧（32 维 padded）投到 expert hidden=720。注意 expert hidden 只有 VLM body 的 75%（960 × 0.75 = 720）, 是论文里的「让 expert 更轻」选择。",
        "trace_name": "model.action_in_proj",
        "in_hint": "action_chunk", "out_hint": "suffix",
    },

    "sinusoidal_time": {
        "type": "leaf",
        "name": "Sinusoidal time emb",
        "sub": "scalar → 720-d",
        "tooltip": "标量 t 编成 720 维正弦位置编码, 跟 transformer 经典 PE 同款。",
        "what": "把每条 sample 的标量 t 编成 720 维向量, 用 transformer 经典的 sin/cos 多频率位置编码。这里编的不是序列位置, 而是「扩散时间」, 让模型从 embedding 就能读懂当前是 t=0.2 还是 t=0.8 等不同阶段。",
        "qualname": "create_sinusoidal_pos_embedding",
        "src": "lerobot/policies/smolvla/modeling_smolvla.py:82",
        "callCount": 1,
        "formula": r"\text{time\_emb}_{2k} = \sin(t/T^{2k/d}), \quad \text{time\_emb}_{2k+1} = \cos(t/T^{2k/d})",
        "shape_html": (
            't: [batch=16] '
            '<span class="arrow">→</span> '
            'time_emb: [batch=16, hidden=720]'
        ),
    },

    "action_time_mlp": {
        "type": "composite",
        "name": "Action+Time MLP",
        "sub": "fuse action & time",
        "tooltip": "[action_emb ‖ time_emb] concat → Linear → SiLU → Linear, 把 t 信息融到每帧的 suffix token 里。",
        "blurb": "把 action embedding 和 time embedding 在 hidden 维 concat 成 (B, 50, 1440), 然后过两层 MLP（中间 SiLU）压回 (B, 50, 720)。这样每个 suffix token 都「知道当前时间」, 后面的 attention 才能根据 t 调整行为。",
        "children": ["action_time_mlp_in", "silu", "action_time_mlp_out"],
    },

    "action_time_mlp_in": {
        "type": "leaf",
        "name": "MLP in",
        "sub": "Linear 1440→720",
        "tooltip": "把 [action ‖ time] (1440=2·720) 压回 hidden=720。",
        "what": "concat 后的 [action_emb ‖ time_emb] 是 1440 维, 这个 Linear 压回 720 维。1M 参数的 Linear, 是 expert 那一侧除了 transformer 块之外最大的可训参数之一。",
        "trace_name": "model.action_time_mlp_in",
        "in_hint": "suffix", "out_hint": "suffix",
    },

    "silu": {
        "type": "leaf",
        "name": "SiLU",
        "sub": "swish activation",
        "tooltip": "\\(\\mathrm{SiLU}(x) = x \\cdot \\sigma(x)\\), 比 ReLU 平滑。LLaMA 系标配。",
        "what": "SiLU（也叫 swish）\\(= x \\cdot \\sigma(x)\\), 比 ReLU 在负值区间有梯度、比 GELU 计算便宜。无可训参数, 不在 trace 里出现独立 forward call。",
        "qualname": "torch.nn.functional.silu",
        "src": "lerobot/policies/smolvla/modeling_smolvla.py:826",
        "callCount": 1,
        "functional": True,
        "shape_html": 'in/out 同 shape: [batch=16, chunk=50, hidden=720]',
        "act_chart": "silu",
        "formula": r"\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}",
    },

    "action_time_mlp_out": {
        "type": "leaf",
        "name": "MLP out",
        "sub": "Linear 720→720",
        "tooltip": "MLP 第二层, 同维度。",
        "what": "保持 720 维的 Linear, 给融合层多一层非线性容量。",
        "trace_name": "model.action_time_mlp_out",
        "in_hint": "suffix", "out_hint": "suffix",
    },

    "vlm_expert": {
        "type": "composite",
        "name": "VLM + Action expert",
        "sub": "vlm_with_expert · 16+16 layers · interleaved CA/SA",
        "tooltip": "SmolVLA 主体。VLM body 处理 prefix（冻结）, Action expert 处理 suffix（训练）, 通过交错的 cross/self attention 让 suffix 拿到 prefix 条件。",
        "blurb": "SmolVLA 的核心。两个 transformer 块并行跑 16 层: VLM body 处理 prefix（视觉/语言/state, hidden=960, 冻结）, lm_expert 处理 suffix（noisy actions, hidden=720, 训练）。论文里每层是 <strong>cross-attention 或 self-attention 二选一</strong>（不是同时都有）, CA 层让 expert Q 去查 VLM 的 K/V, SA 层让 expert token 之间相互 attend（causal mask, 只能看过去）。这种交错设计比纯 SA 或纯 CA 更轻量且效果更好。",
        "children": ["vlm_body", "expert_body"],
    },

    "vlm_body": {
        "type": "composite",
        "name": "VLM body",
        "sub": "16 layers · frozen",
        "tooltip": "SmolVLM 主体, 16 层 Llama-style transformer（GQA + RMSNorm + SwiGLU MLP）。微调全程冻结。",
        "blurb": "SmolVLM 的文本 decoder（被 SmolVLA 改造成 prefix 编码器）。16 层 Llama-style block: RMSNorm → GQA self-attention → residual → RMSNorm → SwiGLU MLP → residual。SmolVLA 把原版 SmolVLM2 的层数砍到一半省推理（论文 ablation）。",
        "children": ["vlm_input_norm", "vlm_self_attn", "vlm_post_attn_norm", "vlm_mlp"],
        "repeat": 16,
    },

    "vlm_input_norm": {
        "type": "leaf", "name": "RMSNorm", "sub": "pre-attention",
        "tooltip": "Llama 系标配, 比 LayerNorm 省一点（只标准化方差, 不减均值）。",
        "what": "RMSNorm 把张量按 <strong>RMS（均方根）</strong> 标准化, 然后乘可学的缩放 \\(\\gamma\\)（没有 bias \\(\\beta\\)）。比 LayerNorm 省两步操作: <strong>不减均值</strong>、<strong>不加偏置</strong>。论文 (Zhang & Sennrich 2019) 证明这两步可以省掉而效果几乎不变。Llama 系全用这个。Vision tower 那边用 LN, 这里换 RMSNorm —— 模型架构混搭, 视觉端遵循 ViT 传统, 语言端跟着 Llama。",
        "trace_name": "model.vlm_with_expert.vlm.model.text_model.layers.0.input_layernorm",
        "in_hint": "prefix", "out_hint": "prefix",
        "formula": r"y_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \varepsilon}} \cdot \gamma_i",
    },

    "vlm_self_attn": {
        "type": "composite", "name": "Self-attention",
        "sub": "GQA · 15 Q heads / 5 KV heads",
        "tooltip": "Grouped-Query Attention: 15 个 Q head 共享 5 组 KV（3:1 比例）, 推理时 KV cache 更小。",
        "blurb": "VLM body 用 grouped-query attention（GQA）省 KV cache: Q 投 960 维（= 15 head × 64）, K/V 只投 320 维（= 5 head × 64）, 每 3 个 Q head 共用一组 KV。SDPA 内部把 KV 沿 head 维度 repeat 3 次再算 attention。",
        "children": ["vlm_q", "vlm_k", "vlm_v", "vlm_sdpa", "vlm_o"],
    },

    "vlm_q": {
        "type": "leaf", "name": "Q proj", "sub": "Linear 960→960 · 15 heads",
        "tooltip": "Q 全 head 投影, 输出 = 15 head × 64 dim = 960。",
        "what": "Q projection, hidden 960 → 960 (15 head × 64 dim)。注意 K/V 维度更小, 这是 GQA 的标志。",
        "trace_name": "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj",
        "in_hint": "prefix", "out_hint": "prefix",
    },
    "vlm_k": {
        "type": "leaf", "name": "K proj", "sub": "Linear 960→320 · 5 heads (GQA)",
        "tooltip": "K 只投到 320 = 5 head × 64, 跟 Q 是 1:3。",
        "what": "GQA 的 K 投影。只投 320 维（5 个 head）, 每个 head 服务 3 个 Q head。省了一半 KV 内存, 在长 seq 推理时是显著的内存节省。",
        "trace_name": "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.k_proj",
        "in_hint": "prefix", "out_hint": "prefix",
        "callout": "<strong>GQA 比例：</strong>VLM body 这里是 15:5 = 3:1（每 3 个 Q head 共享 1 组 KV）。Expert 那边一样的比例。这是 SmolVLA「跨模态共享 KV」的基础 —— K/V 维度对齐才好 concat 做 joint attention。",
    },
    "vlm_v": {
        "type": "leaf", "name": "V proj", "sub": "Linear 960→320 · 5 heads (GQA)",
        "tooltip": "V 维度跟 K 一样, GQA 5 个 head。",
        "what": "V projection, 跟 K 同结构。GQA 让 K 和 V 是同一组 head（共享 head 分组）。",
        "trace_name": "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.v_proj",
        "in_hint": "prefix", "out_hint": "prefix",
    },
    "vlm_sdpa": {
        "type": "leaf", "name": "SDPA", "sub": "F.scaled_dot_product_attention",
        "tooltip": "标准 scaled dot-product attention。KV 内部按 GQA 把 5 head repeat 到 15 再算。",
        "functional": True,
        "what": "VLM body 的 attention 计算。SDPA 内部把 K/V 沿 head 维度复制 3 次（5 → 15）跟 Q 对齐, 再走标准 softmax(QK/√d) V。这里没有像 expert 那侧做 joint attention, 因为 VLM 在这条链路只看 prefix 自己（attention mask 不让 prefix 看 suffix）。",
        "formula": r"\text{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}} + M\right)V",
        "src": "transformers/integrations/sdpa_attention.py:92",
        "callCount": 32,
        "shape_html": (
            'Q: [batch=16, heads=15, seq=177, head_dim=64], '
            'K, V: [batch=16, heads=5→15 (repeat), seq=177, head_dim=64]<br>'
            '<span class="arrow">→</span> [batch=16, heads=15, seq=177, head_dim=64]'
        ),
        "callout": "<strong>注意 mask：</strong>prefix 内部所有 token 互相可见, 但 prefix 不能 attend 到 suffix（attention 2D mask 把 prefix→suffix 那个象限置 0）。这样 prefix 在推理时可以 KV-cache。",
    },
    "vlm_o": {
        "type": "leaf", "name": "Output proj", "sub": "Linear 960→960",
        "tooltip": "Attention 输出投影。",
        "what": "把 multi-head attention 输出合回 hidden=960, 是 attention 块最后一步。",
        "trace_name": "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.o_proj",
        "in_hint": "prefix", "out_hint": "prefix",
    },

    "vlm_post_attn_norm": {
        "type": "leaf", "name": "RMSNorm", "sub": "pre-MLP",
        "tooltip": "MLP 前的 RMSNorm。",
        "what": "MLP 前的 RMSNorm。结构跟 input_layernorm 完全一样, 权重独立。",
        "trace_name": "model.vlm_with_expert.vlm.model.text_model.layers.0.post_attention_layernorm",
        "in_hint": "prefix", "out_hint": "prefix",
    },

    "vlm_mlp": {
        "type": "composite", "name": "SwiGLU MLP",
        "sub": "gate·up → silu·mul → down",
        "tooltip": "Llama 系 SwiGLU FFN: gate × silu(up) 按位乘, 再 down 投回 hidden。",
        "blurb": "SwiGLU FFN（Llama 系标配）。两路并行的 up-projection: gate_proj 和 up_proj 都升到 mlp_dim=2560; 然后 silu(gate) ⊙ up 按位乘融合; 最后 down_proj 投回 960。比标准 GELU FFN 多一路 gate, 多了 ~33% 参数但通常效果更好。",
        "children": ["vlm_gate", "vlm_up", "vlm_silu", "vlm_down"],
    },

    "vlm_gate": {
        "type": "leaf", "name": "gate proj", "sub": "Linear 960→2560",
        "tooltip": "SwiGLU 的 gate 支, 后面会过 SiLU。",
        "what": "SwiGLU 的 gating 支路。升到 mlp_dim=2560 维, 后面跟 SiLU 激活, 控制每个特征单元「开多大」。",
        "trace_name": "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.gate_proj",
        "in_hint": "prefix", "out_hint": "prefix",
    },
    "vlm_up": {
        "type": "leaf", "name": "up proj", "sub": "Linear 960→2560",
        "tooltip": "SwiGLU 的 value 支, 跟 gate 同维度。",
        "what": "SwiGLU 的 value 支, 跟 gate_proj 平行。后面 gate 和 up 按位乘融合, 这是 SwiGLU 区别于 GELU FFN 的关键。",
        "trace_name": "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.up_proj",
        "in_hint": "prefix", "out_hint": "prefix",
    },
    "vlm_silu": {
        "type": "leaf", "name": "SiLU(gate) ⊙ up",
        "sub": "swish + elementwise mul",
        "tooltip": "SwiGLU 的融合: SiLU(gate) 按位乘 up。",
        "what": "SwiGLU 的核心融合操作: gate 过 SiLU 后跟 up 按元素相乘。无可训参数, 不在 trace 里出现独立 forward call（被 mlp 内部当作 functional 处理）。",
        "qualname": "x = act_fn(gate(x)) * up(x)",
        "src": "transformers/models/llama/modeling_llama.py",
        "callCount": 16,
        "functional": True,
        "formula": r"\text{SwiGLU}(x) = \mathrm{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x)",
        "shape_html": '[batch=16, seq=177, mid=2560] (与 gate/up 同维)',
        "act_chart": "silu",
    },
    "vlm_down": {
        "type": "leaf", "name": "down proj", "sub": "Linear 2560→960",
        "tooltip": "MLP 收口投影, 回到 hidden=960。",
        "what": "降维 Linear, 把 SwiGLU 融合后的 2560 投回 hidden 960。是 MLP 块的最后一步, 后面接 residual 进下一层。",
        "trace_name": "model.vlm_with_expert.vlm.model.text_model.layers.0.mlp.down_proj",
        "in_hint": "prefix", "out_hint": "prefix",
    },

    "expert_body": {
        "type": "composite",
        "name": "Action expert",
        "sub": "16 layers · trained",
        "tooltip": "微调时**唯一**被更新的 transformer。结构跟 VLM body 一样（GQA + SwiGLU + RMSNorm）, hidden=720。",
        "blurb": "Action expert 是 SmolVLA 自己加的小 transformer, 跟 VLM body 同结构但 hidden=720（=VLM 的 75%）, FFN mid-dim=2048。<code>train_expert_only=True</code> 让 VLM 冻结、只这一棵子树 + 周边投影头有梯度 —— trace 顶部「~100M trainable」就来自这里。",
        "children": ["exp_input_norm", "exp_self_attn", "exp_post_attn_norm", "exp_mlp"],
        "repeat": 16,
    },

    "exp_input_norm": {
        "type": "leaf", "name": "RMSNorm", "sub": "pre-attention",
        "tooltip": "Expert pre-attn RMSNorm。跟 VLM 一样结构, 但 hidden=720。",
        "what": "Expert 的 pre-attn RMSNorm。结构跟 VLM body 那边完全一致, 只是 hidden 从 960 变成 720。",
        "trace_name": "model.vlm_with_expert.lm_expert.layers.0.input_layernorm",
        "in_hint": "suffix", "out_hint": "suffix",
    },
    "exp_self_attn": {
        "type": "composite", "name": "Joint attention",
        "sub": "Q from suffix · KV joint with prefix",
        "tooltip": "Expert 的 Q 只看自己的 suffix, 但 K/V 是 [VLM 的 prefix KV ; expert 的 suffix KV] 拼起来 —— 这是 SmolVLA prefix→suffix 信息流的唯一通道。",
        "blurb": "<strong>SmolVLA 跟普通双 transformer 的本质差别</strong>。expert 这一侧 attention 算 Q·Kᵀ 时, K 和 V 都不是 expert 自己的, 而是 <code>concat(VLM_KV, expert_KV)</code>。这样 50 个 suffix token 在每一层都能 attend 到 177 个 prefix token + 自己 50 个 = 227 个 token, 视觉/语言信息在每一层都被「灌」进 action 预测。",
        "children": ["exp_q", "exp_k", "exp_v", "exp_sdpa", "exp_o"],
    },

    "exp_q": {
        "type": "leaf", "name": "Q proj", "sub": "Linear 720→960 · 15 heads",
        "tooltip": "Expert Q 投到 960 维（跟 VLM 的 head dim 对齐, 才能 joint attention）。",
        "what": "Expert 的 Q projection。注意输入 hidden 是 720（expert 自己的）, 但输出投到 960（= 15 head × 64）—— 这是为了跟 VLM 的 head_dim 对齐, 让 Q · K_vlm 矩阵乘能算。",
        "trace_name": "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj",
        "in_hint": "suffix", "out_hint": "suffix",
    },
    "exp_k": {
        "type": "leaf", "name": "K proj", "sub": "Linear 720→320 · 5 heads (GQA)",
        "tooltip": "Expert K, GQA 5 head, 跟 VLM 的 K 同维度方便 concat。",
        "what": "Expert 的 K projection, GQA 5 head。维度故意跟 VLM K 完全一致, 这样 K_vlm 和 K_expert 能在 seq 维 concat 起来做 joint attention。",
        "trace_name": "model.vlm_with_expert.lm_expert.layers.0.self_attn.k_proj",
        "in_hint": "suffix", "out_hint": "suffix",
    },
    "exp_v": {
        "type": "leaf", "name": "V proj", "sub": "Linear 720→320 · 5 heads (GQA)",
        "tooltip": "Expert V, 跟 K 同结构。",
        "what": "Expert 的 V projection, 跟 K 平行。设计动机同 K_proj —— 跟 VLM concat 做 joint attention。",
        "trace_name": "model.vlm_with_expert.lm_expert.layers.0.self_attn.v_proj",
        "in_hint": "suffix", "out_hint": "suffix",
    },
    "exp_sdpa": {
        "type": "leaf", "name": "Joint SDPA", "sub": "K/V joint with prefix",
        "tooltip": "Q 是 expert 自己的 50 token, K/V 是 (prefix 177 + suffix 50) = 227 token 的拼接。",
        "functional": True,
        "what": "Expert attention 的关键步骤。Q 是 expert 这 50 个 suffix token, 但 K 和 V 是 <code>cat([K_vlm, K_expert])</code> 和 <code>cat([V_vlm, V_expert])</code>, seq 总长 227。这样每个 action token 都能直接 attend 到所有 prefix tokens（图像 + 语言 + state）, 视觉/语言条件在每一层都被注入 action 预测。",
        "formula": r"\text{attn} = \mathrm{softmax}\!\left(\frac{Q_{\text{exp}} \cdot [K_{\text{vlm}}; K_{\text{exp}}]^\top}{\sqrt{d_k}}\right) [V_{\text{vlm}}; V_{\text{exp}}]",
        "src": "lerobot/policies/smolvla/smolvlm_with_expert.py",
        "callCount": 32,
        "callout": "<strong>这是整个 SmolVLA 的灵魂模块。</strong>VLM 的预训练知识（视觉理解 + 语言条件）就是通过这里被一层一层「灌」进 action expert 的。如果只有 cross-attention 在最后一层, 信息融合不充分; 在每层都 joint attention, 才能让 prefix 的语义渗透到 action chunk。",
        "shape_html": (
            'Q: [batch=16, heads=15, suffix=50, head_dim=64]<br>'
            'K/V: [batch=16, heads=5→15, joint_seq=227, head_dim=64]<br>'
            '<span class="arrow">→</span> [batch=16, heads=15, suffix=50, head_dim=64]'
        ),
    },
    "exp_o": {
        "type": "leaf", "name": "Output proj", "sub": "Linear 960→720",
        "tooltip": "Attention 输出投回 expert hidden=720。",
        "what": "把 joint attention 算出的 (50, 960) 投回 expert 自己的 hidden 720。这一步连接两个不同 hidden 维度的 transformer。",
        "trace_name": "model.vlm_with_expert.lm_expert.layers.0.self_attn.o_proj",
        "in_hint": "suffix", "out_hint": "suffix",
    },

    "exp_post_attn_norm": {
        "type": "leaf", "name": "RMSNorm", "sub": "pre-MLP",
        "tooltip": "Expert 的 pre-MLP RMSNorm。",
        "what": "Expert 的 pre-MLP RMSNorm。跟 input_layernorm 同结构。",
        "trace_name": "model.vlm_with_expert.lm_expert.layers.0.post_attention_layernorm",
        "in_hint": "suffix", "out_hint": "suffix",
    },

    "exp_mlp": {
        "type": "composite", "name": "SwiGLU MLP",
        "sub": "expert FFN · mid=2048",
        "tooltip": "Expert 的 SwiGLU MLP, 中间维 2048（比 VLM 的 2560 小）。",
        "blurb": "跟 VLM MLP 同结构: gate × silu, up 按位乘融合, 再 down。Expert 中间维 2048 比 VLM 的 2560 小 —— 整体参数预算给 expert 留小一点。",
        "children": ["exp_gate", "exp_up", "exp_silu", "exp_down"],
    },

    "exp_gate": {
        "type": "leaf", "name": "gate proj", "sub": "Linear 720→2048",
        "tooltip": "Expert SwiGLU gate。",
        "what": "Expert MLP 的 gate proj, 升到 mlp 中间维 2048。",
        "trace_name": "model.vlm_with_expert.lm_expert.layers.0.mlp.gate_proj",
        "in_hint": "suffix", "out_hint": "suffix",
    },
    "exp_up": {
        "type": "leaf", "name": "up proj", "sub": "Linear 720→2048",
        "tooltip": "Expert SwiGLU value 支。",
        "what": "Expert MLP 的 up proj, 跟 gate 同维度, 后面按位乘。",
        "trace_name": "model.vlm_with_expert.lm_expert.layers.0.mlp.up_proj",
        "in_hint": "suffix", "out_hint": "suffix",
    },
    "exp_silu": {
        "type": "leaf", "name": "SiLU(gate) ⊙ up",
        "sub": "swish + elementwise mul",
        "tooltip": "SwiGLU 融合 (跟 VLM 那边一致)。",
        "what": "Expert MLP 的 SwiGLU 融合操作, 跟 VLM 那边完全相同（只是维度不一样）。",
        "qualname": "x = act_fn(gate(x)) * up(x)",
        "src": "lerobot/policies/smolvla/smolvlm_with_expert.py",
        "callCount": 16,
        "functional": True,
        "shape_html": '[batch=16, chunk=50, mid=2048] (与 gate/up 同维)',
        "act_chart": "silu",
        "formula": r"\text{SwiGLU}(x) = \mathrm{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x)",
    },
    "exp_down": {
        "type": "leaf", "name": "down proj", "sub": "Linear 2048→720",
        "tooltip": "Expert MLP 收口。",
        "what": "Expert MLP 的 down proj, 从中间维 2048 收回 hidden 720。",
        "trace_name": "model.vlm_with_expert.lm_expert.layers.0.mlp.down_proj",
        "in_hint": "suffix", "out_hint": "suffix",
    },

    "action_proj": {
        "type": "leaf",
        "name": "预测速度场",
        "sub": "action_out_proj · Linear 720→32",
        "tooltip": "Expert 输出投回 action 空间, 得到预测的速度场 \\(v_t\\) (不是 action 本身)。",
        "what": "最后一道 Linear, 把 expert hidden 720 投回 max_action_dim=32。<strong>注意输出的不是 action 本身, 而是 flow matching 的速度场 \\(v_t\\)</strong> —— 推理时 ODE 求解器从纯噪声 \\(\\varepsilon\\) 出发, 在每个时间步 \\(t\\) 读这个速度方向把 noisy action 往真实动作推。这是 SmolVLA 跟 \"直接回归 action\" 系（如 ACT, Diffusion Policy 早期）的根本差别。",
        "trace_name": "model.action_out_proj",
        "in_hint": "suffix", "out_hint": "action_chunk",
    },

    "flow_loss": {
        "type": "leaf",
        "name": "Flow loss",
        "sub": "F.mse_loss(v_t, ε − a)",
        "tooltip": "训练目标: 预测速度场 \\(v_t\\) 跟真实速度 \\(u_t = \\varepsilon - a\\) 的 MSE; crop 到 action_dim=6 + 屏蔽 padding 帧。",
        "what": "训练目标。预测的速度场 \\(v_t\\) 跟真实速度 \\(u_t = \\varepsilon - a\\) 算 MSE。然后两层 mask: action chunk 末尾被 episode 截断的 pad 帧不算 loss, 且只在前 action_dim=6 维上累加（pad 到 32 那部分尾巴不算）。v18 first step loss 大约 0.83。",
        "qualname": "torch.nn.functional.mse_loss",
        "src": "lerobot/policies/smolvla/modeling_smolvla.py:892",
        "callCount": 1,
        "functional": True,
        "formula": r"\mathcal{L} = \frac{1}{N_{\text{valid}} \cdot 6} \sum_{i,k,d<6,\, \neg\text{pad}_{i,k}} \|v_{t,i,k,d} - u_{t,i,k,d}\|^2",
        "shape_html": (
            'v_t, u_t: [batch=16, chunk=50, action_dim=32]<br>'
            '<span class="arrow">→</span> crop d 维到 6 + 屏蔽 pad 帧 <span class="arrow">→</span> scalar loss'
        ),
    },

    "backward_step": {
        "type": "leaf",
        "name": "Backward + step",
        "sub": "350 grads · AdamW · sentinel",
        "tooltip": "loss.backward() → clip_grad_norm → AdamW.step → sentinel 抛 TraceComplete, trace 结束。",
        "what": "loss.backward() 沿 autograd 图反传, VLM body 跟视觉塔的参数 requires_grad=False, 梯度算到它们就停; 只有 expert + 几个投影头收到非零 grad。trace 用 tensor-level register_hook 抓到 350 个 backward 事件。AdamW 一步更新后, instrument.py 的 step_post_hook 抛 TraceComplete, 训练循环立即退出。",
        "qualname": "accelerator.backward + torch.optim.AdamW.step",
        "src": "lerobot/scripts/lerobot_train.py:127-141",
        "callCount": 1,
        "formula": r"\theta \leftarrow \theta - \alpha \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}",
        "shape_html": 'loss: scalar (≈ 0.83 at step #1) → grads on ~100M trainable params',
    },
}
