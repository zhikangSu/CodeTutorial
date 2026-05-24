"""render_html.py — SmolVLA trace JSON → 可交互架构图 HTML（v4）。

实现遵循用户提供的 UX spec：
- 首屏一张方块架构图，左→右数据流
- composite 节点点击 = 主图替换为子节点
- leaf 节点点击 = 右侧抽屉详情
- 重复层（VLM 16 / Expert 16 / Vision 12）合并为单节点 + ×N badge
- shape 用语义维度名（[batch=16, seq=177, hidden=960]）
- Esc 收抽屉 / 回 breadcrumb

Tree 是手设计的概念层级，真实 shape / src / callCount 从 trace JSON 拿。

用法:
    python render_html.py <trace.json> [<output.html>]
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any


# ===========================================================================
# trace JSON helpers
# ===========================================================================
def load_trace(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def find_tensor_shape(v: Any) -> list[int] | None:
    if isinstance(v, dict):
        if v.get("_type") == "Tensor":
            return v.get("shape")
        items = v.get("items")
        if isinstance(items, list):
            for x in items:
                r = find_tensor_shape(x)
                if r:
                    return r
        elif isinstance(items, dict):
            for x in items.values():
                r = find_tensor_shape(x)
                if r:
                    return r
    return None


def find_tensor_dtype(v: Any) -> str | None:
    if isinstance(v, dict):
        if v.get("_type") == "Tensor":
            return v.get("dtype")
        items = v.get("items")
        if isinstance(items, list):
            for x in items:
                r = find_tensor_dtype(x)
                if r:
                    return r
        elif isinstance(items, dict):
            for x in items.values():
                r = find_tensor_dtype(x)
                if r:
                    return r
    return None


def pair_events_by_name(trace: dict) -> dict[str, list[dict]]:
    mt = trace["module_tree"]
    by_name: dict[str, list[dict]] = {}
    stack: list[dict] = []
    for e in trace["module_events"]:
        if e["phase"] == "forward_pre":
            stack.append(e)
        elif e["phase"] == "forward_post":
            if not stack:
                continue
            pre = stack.pop()
            if pre["module_id"] != e["module_id"]:
                continue
            mt_entry = mt.get(str(pre["module_id"])) or mt.get(pre["module_id"])
            if not mt_entry:
                continue
            by_name.setdefault(mt_entry["name"], []).append({
                "inputs": pre["inputs"],
                "outputs": e["outputs"],
                "duration_ms": e.get("duration_ms"),
                "src_file": pre.get("src_file"),
                "src_line": pre.get("src_line"),
                "qualname": pre.get("qualname"),
                "num_params_total": mt_entry.get("num_params_total"),
            })
    return by_name


# ===========================================================================
# Semantic dimension labeling
# ===========================================================================
def fmt_shape(shape: list[int] | None, hint: str = "", dtype: str = "") -> str:
    if shape is None:
        return "—"
    sd = _dim_names(shape, hint)
    inner = ", ".join(
        f'<span class="dim">{name}</span>=<span class="dim-val">{v}</span>'
        for name, v in zip(sd, shape)
    )
    suffix = f' <span class="dtype">{dtype}</span>' if dtype else ""
    return f'[{inner}]{suffix}'


def _dim_names(shape: list[int], hint: str = "") -> list[str]:
    n = len(shape)
    if n == 4:
        return ["batch", "c", "h", "w"]
    if n == 3:
        if hint == "action_chunk":
            return ["batch", "chunk", "action_dim"]
        if hint == "suffix":
            return ["batch", "chunk", "hidden"]
        if hint == "vision_tokens":
            return ["batch", "tokens", "hidden"]
        if hint == "connector_in":
            return ["batch", "tokens", "hidden_vis"]
        if hint == "connector_out":
            return ["batch", "tokens", "hidden"]
        if hint == "prefix":
            return ["batch", "seq", "hidden"]
        return ["batch", "seq", "hidden"]
    if n == 2:
        if hint == "state_padded":
            return ["batch", "state_dim"]
        if hint == "state_proj_out":
            return ["batch", "hidden"]
        if hint == "lang_tokens":
            return ["batch", "seq_lang"]
        if hint == "time":
            return ["batch", "hidden"]
        return ["batch", "dim"]
    if n == 1:
        return ["batch"]
    if n == 5:
        return ["batch", "t", "c", "h", "w"]
    return [f"d{i}" for i in range(n)]


def fmt_arrow(in_shape, out_shape, in_hint="", out_hint="", in_dtype="", out_dtype=""):
    return (
        fmt_shape(in_shape, in_hint, in_dtype)
        + ' <span class="arrow">→</span> '
        + fmt_shape(out_shape, out_hint, out_dtype)
    )


# Relative-path → absolute-path mapping for vscode:// links.
# 调用方可以通过环境变量 SMOLVLA_TRACE_PATH_MAP="prefix1=/abs1;prefix2=/abs2" 覆盖。
PATH_MAP = {
    "lerobot/": "/data/users/szk/repos/lerobot/src/lerobot/",
    "transformers/": "/data/users/szk/miniconda3/envs/lerobot/lib/python3.12/site-packages/transformers/",
    "torch/": "/data/users/szk/miniconda3/envs/lerobot/lib/python3.12/site-packages/torch/",
}

# 显式 src override: leaf node id → SmolVLA 源码声明位置 (相对路径:行号)
# trace 抓的是 forward 的运行位置（如 torch/nn/modules/linear.py:53），对教学没用。
# 这里 override 成 SmolVLA / SmolVLM / Llama 里"模块被声明出来的那一行"。
SMOLVLA_DECL_SRC: dict[str, str] = {
    # SmolVLA-specific projections (lerobot 源码)
    "state_proj":         "lerobot/policies/smolvla/modeling_smolvla.py:653",
    "action_in_proj":     "lerobot/policies/smolvla/modeling_smolvla.py:656",
    "action_out_proj":    "lerobot/policies/smolvla/modeling_smolvla.py:657",
    "action_proj":        "lerobot/policies/smolvla/modeling_smolvla.py:657",
    "action_time_mlp_in": "lerobot/policies/smolvla/modeling_smolvla.py:659",
    "action_time_mlp_out":"lerobot/policies/smolvla/modeling_smolvla.py:662",
    "lang_embed":         "lerobot/policies/smolvla/smolvlm_with_expert.py:206",
    # SmolVLM2 vision tower (transformers SmolVLM 源码)
    "patch_embed":            "transformers/models/smolvlm/modeling_smolvlm.py:84",
    "vis_norm1":              "transformers/models/smolvlm/modeling_smolvlm.py:245",
    "vis_norm2":              "transformers/models/smolvlm/modeling_smolvlm.py:245",
    "vis_q":                  "transformers/models/smolvlm/modeling_smolvlm.py:166",
    "vis_k":                  "transformers/models/smolvlm/modeling_smolvlm.py:166",
    "vis_v":                  "transformers/models/smolvlm/modeling_smolvlm.py:166",
    "vis_out":                "transformers/models/smolvlm/modeling_smolvlm.py:166",
    "vis_fc1":                "transformers/models/smolvlm/modeling_smolvlm.py:230",
    "vis_fc2":                "transformers/models/smolvlm/modeling_smolvlm.py:230",
    "vis_gelu":               "transformers/models/smolvlm/modeling_smolvlm.py:230",
    "vision_post_layernorm":  "transformers/models/smolvlm/modeling_smolvlm.py:318",
    "connector":              "transformers/models/smolvlm/modeling_smolvlm.py:431",
    # VLM body (Llama 实现, 在 transformers 里), SmolVLA 在 smolvlm_with_expert.py 里组合调用
    "vlm_input_norm":     "transformers/models/llama/modeling_llama.py:53",
    "vlm_post_attn_norm": "transformers/models/llama/modeling_llama.py:53",
    "vlm_q":              "transformers/models/llama/modeling_llama.py:238",
    "vlm_k":              "transformers/models/llama/modeling_llama.py:238",
    "vlm_v":              "transformers/models/llama/modeling_llama.py:238",
    "vlm_o":              "transformers/models/llama/modeling_llama.py:238",
    "vlm_gate":           "transformers/models/llama/modeling_llama.py:177",
    "vlm_up":             "transformers/models/llama/modeling_llama.py:177",
    "vlm_down":           "transformers/models/llama/modeling_llama.py:177",
    # Expert body. K/V 投影被 SmolVLA 在 smolvlm_with_expert.py 里 override 过, 指向那里
    "exp_input_norm":     "transformers/models/llama/modeling_llama.py:53",
    "exp_post_attn_norm": "transformers/models/llama/modeling_llama.py:53",
    "exp_q":              "transformers/models/llama/modeling_llama.py:238",
    "exp_k":              "lerobot/policies/smolvla/smolvlm_with_expert.py:125",
    "exp_v":              "lerobot/policies/smolvla/smolvlm_with_expert.py:130",
    "exp_o":              "transformers/models/llama/modeling_llama.py:238",
    "exp_gate":           "transformers/models/llama/modeling_llama.py:177",
    "exp_up":             "transformers/models/llama/modeling_llama.py:177",
    "exp_down":           "transformers/models/llama/modeling_llama.py:177",
    # joint attention forward 逻辑（SmolVLA 自己的 SDPA 调用入口）
    "exp_sdpa":           "lerobot/policies/smolvla/smolvlm_with_expert.py:209",
}


def vscode_link(rel_src: str) -> str | None:
    """从 'lerobot/policies/smolvla/modeling_smolvla.py:653' 构造 vscode:// 链接。"""
    if not rel_src or rel_src == "—" or ":" not in rel_src:
        return None
    rel_path, _, lno = rel_src.rpartition(":")
    if not lno or (not lno.isdigit() and "-" not in lno):
        return None
    # 如果已经是绝对路径，直接用
    if rel_path.startswith("/"):
        return f"vscode://file{rel_path}:{lno}"
    for prefix, abs_prefix in PATH_MAP.items():
        if rel_path.startswith(prefix):
            return f"vscode://file{abs_prefix}{rel_path[len(prefix):]}:{lno}"
    return None


def activation_svg(kind: str) -> str:
    """生成 GELU / SiLU 等激活函数的小 SVG 曲线, 嵌进 drawer 帮助直观看图。"""
    import math
    if kind == "gelu":
        f = lambda x: 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
        title = "GELU(x) = x · Φ(x)"
    elif kind == "silu":
        f = lambda x: x / (1 + math.exp(-x)) if x > -50 else 0.0
        title = "SiLU(x) = x · σ(x)"
    else:
        return ""
    xs = [-4 + i * 0.2 for i in range(41)]
    ys = [f(x) for x in xs]
    # SVG y axis inverted; map y -> -y for upward curve
    pts = " ".join(f"{x:.2f},{-y:.3f}" for x, y in zip(xs, ys))
    axes = (
        '<line x1="-4" y1="0" x2="4" y2="0" stroke="#ccc" stroke-width="0.03"/>'
        '<line x1="0" y1="-4" x2="0" y2="1" stroke="#ccc" stroke-width="0.03"/>'
        '<text x="3.6" y="0.4" font-size="0.32" fill="#888" font-family="JetBrains Mono, monospace">x</text>'
        '<text x="0.12" y="-3.6" font-size="0.32" fill="#888" font-family="JetBrains Mono, monospace">y</text>'
    )
    return (
        f'<div class="act-curve">'
        f'<div class="act-curve-title">{title}</div>'
        f'<svg viewBox="-4 -4 8 5" preserveAspectRatio="xMidYMid meet" class="act-curve-svg">'
        f'{axes}'
        f'<polyline points="{pts}" fill="none" stroke="#C7472A" stroke-width="0.06" stroke-linejoin="round"/>'
        f'</svg>'
        f'</div>'
    )


# ===========================================================================
# Hand-designed conceptual tree (annotations + structure)
# Real shape / src / callCount filled in by enrich_with_trace().
# ===========================================================================
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
        # 兼容老的 children 字段，drill-in 时还按列出顺序展开
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


# ===========================================================================
# Enrich tree with real trace data (shapes, src, call counts)
# ===========================================================================
def enrich_with_trace(tree: dict, calls_by_name: dict[str, list[dict]]) -> dict:
    enriched: dict[str, Any] = {}
    for nid, node in tree.items():
        node = dict(node)
        # 应用 SmolVLA 声明位置 override（总是覆盖, 因为 trace 抓到的 src 通常是 torch 内部 forward）
        if nid in SMOLVLA_DECL_SRC:
            node["src"] = SMOLVLA_DECL_SRC[nid]
        if node.get("type") == "leaf" and node.get("trace_name"):
            tn = node["trace_name"]
            calls_all: list[dict] = []
            if ".layers.0." in tn:
                pattern = re.sub(r"\.layers\.0\.", r".layers.\\d+.", re.escape(tn))
                regex = re.compile("^" + pattern + "$")
                for name, c in calls_by_name.items():
                    if regex.match(name):
                        calls_all.extend(c)
            else:
                calls_all = calls_by_name.get(tn, [])

            if calls_all:
                first = calls_all[0]
                in_shape = find_tensor_shape(first["inputs"])
                out_shape = find_tensor_shape(first["outputs"])
                in_dtype = find_tensor_dtype(first["inputs"]) or ""
                out_dtype = find_tensor_dtype(first["outputs"]) or ""
                node["_shape_html"] = fmt_arrow(
                    in_shape, out_shape,
                    node.get("in_hint", ""), node.get("out_hint", ""),
                    in_dtype, out_dtype,
                )
                node["_qualname"] = first.get("qualname", "")
                node["_callCount"] = len(calls_all)
                # src 选择:
                #   静态 node['src'] 优先 (指向 SmolVLA 源码的声明位置)
                #   否则用 trace 抓到的 src_file (forward 实际执行的位置, 如 torch/nn/modules/linear.py)
                static_src = node.get("src")
                if static_src and static_src != "—":
                    node["_src"] = static_src
                    node["_src_link"] = vscode_link(static_src)
                    # 也保留 trace 抓到的 forward 位置作为副信息
                    abs_src = first.get("src_file") or ""
                    line_no = first.get('src_line', '?')
                    if abs_src:
                        trace_display = abs_src
                        for marker in ("lerobot/", "transformers/", "torch/"):
                            if marker in trace_display:
                                trace_display = trace_display[trace_display.index(marker):]
                                break
                        else:
                            if "/site-packages/" in trace_display:
                                trace_display = trace_display.split("/site-packages/")[-1]
                        node["_trace_src"] = f"{trace_display}:{line_no}"
                        node["_trace_src_link"] = f"vscode://file{abs_src}:{line_no}" if abs_src.startswith("/") else None
                else:
                    # 无静态 src, fallback 到 trace
                    abs_src = first.get("src_file") or ""
                    line_no = first.get('src_line', '?')
                    src_display = abs_src
                    for marker in ("lerobot/", "transformers/", "torch/"):
                        if marker in src_display:
                            src_display = src_display[src_display.index(marker):]
                            break
                    else:
                        if "/site-packages/" in src_display:
                            src_display = src_display.split("/site-packages/")[-1]
                    node["_src"] = f"{src_display}:{line_no}"
                    if abs_src and abs_src.startswith("/"):
                        node["_src_link"] = f"vscode://file{abs_src}:{line_no}"
            else:
                node["_shape_html"] = node.get("shape_html", "—")
                node["_qualname"] = node.get("qualname", "")
                node["_src"] = node.get("src", "—")
                node["_callCount"] = node.get("callCount", 1)
                node["_src_link"] = vscode_link(node["_src"])
        elif node.get("type") == "leaf":
            node["_shape_html"] = node.get("shape_html", "—")
            node["_qualname"] = node.get("qualname", "")
            node["_src"] = node.get("src", "—")
            node["_callCount"] = node.get("callCount", 1)
            node["_src_link"] = vscode_link(node["_src"])
        # 激活函数曲线
        if node.get("act_chart"):
            node["_act_chart_svg"] = activation_svg(node["act_chart"])
        enriched[nid] = node
    return enriched


# ===========================================================================
# CSS (adapted from user's UX spec) + KaTeX + fonts
# ===========================================================================
CSS = r"""
:root {
  --bg: #FAF7F0;
  --surface: #FFFFFF;
  --surface-alt: #F5F2EA;
  --surface-warm: #EFEAE0;
  --text: #1F1B16;
  --text-2: #6B6354;
  --text-3: #9A9080;
  --border: #E5DDD0;
  --border-strong: #D4C9B8;
  --accent: #C7472A;
  --accent-2: #9A3719;
  --accent-soft: #F4D9CB;
  --accent-soft-2: #FBE9DF;
  --focus: #2B6CB0;
  --shadow-1: 0 1px 2px rgba(50, 30, 10, 0.04);
  --shadow-2: 0 4px 16px rgba(50, 30, 10, 0.08);
  --shadow-3: 0 8px 32px rgba(50, 30, 10, 0.12);
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: 'DM Sans', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
.container { max-width: 1280px; margin: 0 auto; padding: 32px 32px 96px; }
h1 {
  font-family: 'Bricolage Grotesque', serif;
  font-weight: 700;
  font-size: 28px;
  line-height: 1.15;
  margin: 0 0 6px;
  letter-spacing: -0.015em;
}
.lede { color: var(--text-2); font-size: 13px; margin: 0 0 24px; font-family: 'JetBrains Mono', monospace; }
.lede b { color: var(--text); }
.lede .dot { color: var(--text-3); margin: 0 8px; }

.canvas-wrap {
  background: var(--surface);
  border: 0.5px solid var(--border);
  border-radius: 14px;
  padding: 22px 24px 26px;
  position: relative;
  min-height: 380px;
}
.breadcrumb {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 13px;
  color: var(--text-2);
  margin-bottom: 8px;
  flex-wrap: wrap;
  min-height: 28px;
}
.breadcrumb a {
  color: var(--text-2);
  text-decoration: none;
  cursor: pointer;
  padding: 3px 8px;
  border-radius: 5px;
  transition: background 0.12s, color 0.12s;
}
.breadcrumb a:hover { background: var(--surface-alt); color: var(--text); }
.breadcrumb .current { color: var(--text); font-weight: 500; padding: 3px 8px; }
.breadcrumb .sep { color: var(--text-3); padding: 0 1px; user-select: none; }
.breadcrumb .meta {
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-3);
  letter-spacing: 0.2px;
}

.blurb {
  font-size: 13.5px;
  color: var(--text-2);
  margin: 10px 0 22px;
  padding: 12px 16px;
  background: var(--surface-alt);
  border-left: 3px solid var(--accent);
  border-radius: 0 6px 6px 0;
  line-height: 1.65;
}
.blurb:empty { display: none; }
.blurb code {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  background: var(--surface);
  padding: 1px 5px;
  border-radius: 3px;
  border: 0.5px solid var(--border);
}
.blurb strong { color: var(--accent-2); }

.canvas { display: flex; flex-direction: column; align-items: center; gap: 6px; min-height: 220px; padding: 12px 0; }
.row {
  display: flex;
  align-items: stretch;
  gap: 2px;
  flex-wrap: wrap;
  justify-content: center;
  max-width: 100%;
}

/* ---- stages layout (Y-shape topology) ---- */
.stages-wrap {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 40px;
  padding: 24px 0 36px;
  width: 100%;
  max-width: 1000px;
}
.arrows-svg {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
  z-index: 0;
}
.arrows-svg .arrow-line { pointer-events: none; }
.stage {
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: stretch;
  gap: 48px;
  z-index: 2;
  position: relative;
}
.stage.training-only .node {
  border-style: dashed;
  opacity: 0.72;
}
.stage.training-only .node.composite {
  background: var(--surface-alt);
}
.node-slot { display: flex; align-items: stretch; }
.edge-labels {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 1;
}
.edge-label {
  position: absolute;
  transform: translate(-50%, 0);
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  color: var(--text-2);
  background: var(--bg);
  padding: 2px 8px;
  border-radius: 3px;
  border: 0.5px solid var(--border);
  white-space: nowrap;
  pointer-events: none;
  box-shadow: var(--shadow-1);
}
.node.is-heart {
  min-width: 220px;
  padding: 16px 20px;
  border-width: 1.5px;
  background: var(--accent);
  color: white;
  border-color: var(--accent-2);
}
.node.is-heart .node-title { color: white; font-size: 16px; font-weight: 700; }
.node.is-heart .node-sub { color: rgba(255, 255, 255, 0.78); font-size: 11.5px; }
.node.is-heart::after { color: rgba(255, 255, 255, 0.9); }
.shape-strip {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-2);
  padding: 8px 14px;
  letter-spacing: 0.2px;
  text-align: center;
  line-height: 1.6;
  background: var(--surface-alt);
  border-radius: 4px;
  margin: 4px 0;
}
.shape-strip .mono { font-family: 'JetBrains Mono', monospace; background: var(--surface); padding: 1px 6px; border-radius: 3px; border: 0.5px solid var(--border); color: var(--text); }
.arrow-h { color: var(--text-3); font-size: 14px; padding: 0 4px; user-select: none; display: flex; align-items: center; }
.arrow-v { color: var(--text-3); font-size: 18px; line-height: 1; padding: 2px 0; }

.node {
  position: relative;
  background: var(--surface);
  border: 0.5px solid var(--border-strong);
  border-radius: 8px;
  padding: 10px 14px;
  min-width: 140px;
  max-width: 220px;
  text-align: center;
  cursor: pointer;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
  user-select: none;
  overflow-wrap: break-word;
  word-break: break-word;
  hyphens: auto;
}
.node-title { line-height: 1.35; }
.node-sub { line-height: 1.4; }
.node:hover {
  border-color: var(--accent);
  transform: translateY(-1px);
  box-shadow: var(--shadow-2);
  z-index: 5;
}
.node:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.node-title { font-size: 13px; font-weight: 500; color: var(--text); }
.node-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  color: var(--text-3);
  margin-top: 3px;
}
.node.composite { background: var(--accent-soft-2); border-color: var(--accent-soft); }
.node.composite .node-sub { color: var(--accent-2); opacity: 0.85; }
.node.composite::after {
  content: '\25B8';
  position: absolute;
  top: 3px;
  right: 7px;
  font-size: 11px;
  color: var(--accent);
  opacity: 0.7;
}
.node.functional { border-style: dashed; background: #FFF5D6; border-color: #EBDCA5; }
.node.functional .node-sub { color: #7A5A0A; }

.repeat-badge {
  display: inline-block;
  background: var(--accent);
  color: white;
  font-family: 'JetBrains Mono', monospace;
  font-size: 9.5px;
  padding: 1px 5px;
  border-radius: 8px;
  margin-left: 4px;
  vertical-align: middle;
  letter-spacing: 0.3px;
}

.tooltip {
  position: absolute;
  bottom: calc(100% + 10px);
  left: 50%;
  transform: translateX(-50%);
  background: var(--text);
  color: #FAF7F0;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 11.5px;
  line-height: 1.5;
  width: 220px;
  z-index: 100;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.12s ease, transform 0.12s ease;
  box-shadow: var(--shadow-2);
}
.tooltip::after {
  content: '';
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 5px solid transparent;
  border-top-color: var(--text);
}
.node:hover .tooltip { opacity: 1; transform: translateX(-50%) translateY(-2px); }

.legend {
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  margin-top: 28px;
  padding-top: 16px;
  border-top: 0.5px solid var(--border);
  font-size: 11.5px;
  color: var(--text-2);
}
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.swatch.composite { background: var(--accent-soft-2); border: 0.5px solid var(--accent-soft); }
.swatch.leaf { background: var(--surface); border: 0.5px solid var(--border-strong); }
.swatch.functional { background: #FFF5D6; border: 0.5px dashed #EBDCA5; }
.legend kbd {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  padding: 1px 6px; border: 0.5px solid var(--border-strong); border-radius: 3px;
  background: var(--surface); color: var(--text-2);
}

.drawer-backdrop {
  position: fixed; inset: 0;
  background: rgba(31, 27, 22, 0.18);
  opacity: 0; pointer-events: none;
  transition: opacity 0.18s ease;
  z-index: 90;
}
.drawer-backdrop.open { opacity: 1; pointer-events: auto; }
.drawer {
  position: fixed; top: 0; right: 0; bottom: 0;
  width: 480px; max-width: 92vw;
  background: var(--bg);
  border-left: 0.5px solid var(--border);
  transform: translateX(100%);
  transition: transform 0.24s cubic-bezier(0.2, 0.8, 0.2, 1);
  z-index: 100;
  overflow-y: auto;
  padding: 24px 28px 56px;
  box-shadow: -8px 0 32px rgba(50, 30, 10, 0.08);
}
.drawer.open { transform: translateX(0); }
.drawer-close {
  position: absolute; top: 14px; right: 16px;
  background: transparent; border: none;
  font-size: 22px; cursor: pointer;
  color: var(--text-2); padding: 4px 10px;
  border-radius: 6px; line-height: 1;
}
.drawer-close:hover { background: var(--surface-alt); color: var(--text); }
.drawer h2 {
  font-family: 'Bricolage Grotesque', serif;
  font-size: 22px; font-weight: 700;
  margin: 0 0 4px; letter-spacing: -0.01em;
}
.drawer .qualname {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: var(--text-2);
  margin: 0 0 14px; word-break: break-all;
}
.drawer .qualname .call-count {
  display: inline-block;
  background: var(--accent-soft-2);
  color: var(--accent-2);
  padding: 1px 7px; border-radius: 4px;
  font-size: 11px; margin-left: 6px; letter-spacing: 0.3px;
}
.drawer .badge-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 18px; }
.drawer .badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; padding: 2px 8px;
  border-radius: 4px; background: var(--surface-alt);
  color: var(--text-2); border: 0.5px solid var(--border);
}
.drawer .badge.functional { background: #FFF5D6; color: #7A5A0A; border-color: #EBDCA5; }
.drawer h3 {
  font-size: 11.5px; font-weight: 500;
  color: var(--text-2); text-transform: uppercase;
  letter-spacing: 0.08em; margin: 22px 0 8px;
}
.drawer p {
  font-size: 14px; line-height: 1.65;
  color: var(--text); margin: 0 0 10px;
}
.drawer p em { color: var(--accent-2); font-style: italic; }
.shape-box {
  background: var(--surface);
  border: 0.5px solid var(--border);
  border-radius: 6px;
  padding: 10px 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px; line-height: 1.75;
  margin: 0; word-break: break-word;
}
.shape-box .arrow { color: var(--text-3); margin: 0 6px; }
.shape-box .dim { color: var(--accent-2); font-weight: 500; }
.shape-box .dim-val { color: var(--text); }
.shape-box .dtype { color: var(--text-3); font-style: italic; margin-left: 6px; font-size: 11px; }
.formula {
  background: var(--surface);
  border: 0.5px solid var(--border);
  border-radius: 6px;
  padding: 14px 16px;
  margin: 0; text-align: center;
}
.formula .katex { font-size: 1.05em; }
.callout {
  background: var(--accent-soft-2);
  border-left: 3px solid var(--accent);
  padding: 10px 14px;
  margin: 12px 0 0;
  font-size: 13px; color: var(--text);
  line-height: 1.6;
  border-radius: 0 6px 6px 0;
}
.callout strong { color: var(--accent-2); }
.callout code {
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  background: var(--surface); padding: 1px 5px; border-radius: 3px;
  border: 0.5px solid var(--border);
}
.src-link {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: var(--text-2);
  word-break: break-all;
}
.src-link a { color: var(--accent-2); text-decoration: underline; text-underline-offset: 2px; }
.src-link a:hover { color: var(--accent); }
.src-link-trace {
  margin-top: 4px;
  font-size: 10.5px;
  color: var(--text-3);
}
.src-link-trace a { color: var(--text-3); }

.act-curve {
  margin: 12px 0;
}
.act-curve-title {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11.5px;
  color: var(--text-2);
  margin-bottom: 4px;
}
.act-curve-svg {
  width: 100%;
  max-width: 320px;
  height: 110px;
  background: var(--surface);
  border: 0.5px solid var(--border);
  border-radius: 6px;
  padding: 6px;
  display: block;
}

@media (max-width: 720px) {
  .container { padding: 18px 14px 64px; }
  .drawer { width: 100vw; }
}
"""

FONT_AND_KATEX = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Bricolage+Grotesque:wght@500;600;700&'
    'family=DM+Sans:wght@400;500;700&'
    'family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'
    '<link rel="stylesheet" '
    'href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css" crossorigin="anonymous">'
    '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js" crossorigin="anonymous"></script>'
    '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js" '
    'crossorigin="anonymous" '
    "onload=\"renderMathInElement(document.body, {delimiters: ["
    "{left: '$$', right: '$$', display: true}, "
    "{left: '\\\\(', right: '\\\\)', display: false}"
    "]});\"></script>"
)


JS_TEMPLATE = r"""
const TREE = __TREE_JSON__;

const state = {
  path: ['pipeline'],
  drawerOpen: null,
};

function renderBreadcrumb() {
  const el = document.getElementById('breadcrumb');
  const segments = state.path.map((id, idx) => {
    const isLast = idx === state.path.length - 1;
    const name = TREE[id].name;
    if (isLast) return '<span class="current">' + name + '</span>';
    return '<a data-bc-idx="' + idx + '">' + name + '</a><span class="sep">▸</span>';
  }).join('');
  el.innerHTML = segments + '<span class="meta">batch=16 · bf16 · v18 single-GPU</span>';
  el.querySelectorAll('a[data-bc-idx]').forEach(a => {
    a.addEventListener('click', () => {
      state.path = state.path.slice(0, parseInt(a.dataset.bcIdx, 10) + 1);
      render();
    });
  });
}

function renderBlurb() {
  const el = document.getElementById('blurb');
  const cur = TREE[state.path[state.path.length - 1]];
  let html = '';
  if (state.path.length === 1 && cur.story_line) html = cur.story_line;
  else if (state.path.length > 1 && cur.blurb) html = cur.blurb;
  el.innerHTML = html;
  if (html && window.renderMathInElement) {
    renderMathInElement(el, {
      delimiters: [{left: '$$', right: '$$', display: true}, {left: '\\(', right: '\\)', display: false}]
    });
  }
}

function renderNode(id, opts) {
  opts = opts || {};
  const n = TREE[id];
  if (!n) return '<div class="node leaf">[missing: ' + id + ']</div>';
  const cls = ['node', n.type];
  if (n.functional) cls.push('functional');
  if (opts.heart || n.is_heart) cls.push('is-heart');
  if (opts.training_only) cls.push('training-only');
  const repeat = n.repeat ? '<span class="repeat-badge">×' + n.repeat + '</span>' : '';
  return '<div class="' + cls.join(' ') + '" data-id="' + id + '" tabindex="0" role="button" aria-label="' + n.name + '">'
    + '<div class="node-title">' + n.name + repeat + '</div>'
    + (n.sub ? '<div class="node-sub">' + n.sub + '</div>' : '')
    + (n.tooltip ? '<div class="tooltip">' + n.tooltip + '</div>' : '')
    + '</div>';
}

function renderRow(ids) {
  return ids.map((id, i) => (i > 0 ? '<span class="arrow-h">→</span>' : '') + renderNode(id)).join('');
}

function renderStages(stages) {
  // Vertical layout: each stage is a row; between rows draw SVG arrows + optional edge labels.
  let html = '';
  stages.forEach((stage, i) => {
    const stageClass = ['stage'];
    if (stage.training_only) stageClass.push('training-only');
    html += '<div class="' + stageClass.join(' ') + '" data-stage-idx="' + i + '">';
    stage.nodes.forEach(nodeId => {
      const heart = !!stage.is_heart;
      html += '<div class="node-slot" data-id="' + nodeId + '">' +
              renderNode(nodeId, {heart: heart, training_only: stage.training_only}) + '</div>';
    });
    html += '</div>';
    // outgoing labels container (positions resolved by JS after layout)
    if (stage.outgoing_labels) {
      html += '<div class="edge-labels" data-from-stage="' + i + '">';
      Object.entries(stage.outgoing_labels).forEach(([fromId, label]) => {
        html += '<div class="edge-label" data-from="' + fromId + '">' + label + '</div>';
      });
      html += '</div>';
    }
  });
  return '<div class="stages-wrap">'
    + '<svg class="arrows-svg" id="arrows-svg" xmlns="http://www.w3.org/2000/svg">'
    +   '<defs><marker id="arrowhead" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">'
    +     '<path d="M0,0 L8,5 L0,10 L2,5 Z" fill="#9A9080" /></marker></defs>'
    + '</svg>'
    + html
    + '</div>';
}

function renderCanvas() {
  const canvas = document.getElementById('canvas');
  const cur = TREE[state.path[state.path.length - 1]];

  if (cur.layout === 'stages' && cur.stages) {
    canvas.innerHTML = renderStages(cur.stages);
    canvas.querySelectorAll('.node').forEach(el => attachNodeHandlers(el));
    // Compute arrow positions after browser layout
    requestAnimationFrame(() => { drawStageArrows(cur.stages); });
    // Redraw on window resize
    if (!window._smolvlaArrowsBound) {
      window._smolvlaArrowsBound = true;
      window.addEventListener('resize', () => {
        const c = TREE[state.path[state.path.length - 1]];
        if (c.layout === 'stages' && c.stages) drawStageArrows(c.stages);
      });
    }
  } else {
    const children = cur.children || [];
    canvas.innerHTML = '<div class="row">' + renderRow(children) + '</div>';
    canvas.querySelectorAll('.node').forEach(el => attachNodeHandlers(el));
  }
}

function attachNodeHandlers(el) {
  el.addEventListener('click', () => handleNodeClick(el.dataset.id));
  el.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleNodeClick(el.dataset.id); }
  });
}

function drawStageArrows(stages) {
  const svg = document.getElementById('arrows-svg');
  if (!svg) return;
  const wrap = svg.parentElement;
  const wrapRect = wrap.getBoundingClientRect();
  // Size svg to overlay full wrap
  svg.setAttribute('width', wrapRect.width);
  svg.setAttribute('height', wrapRect.height);
  svg.setAttribute('viewBox', '0 0 ' + wrapRect.width + ' ' + wrapRect.height);
  // Clear old paths but keep defs
  Array.from(svg.querySelectorAll('path.arrow-line, text.shape-label')).forEach(e => e.remove());

  const stageEls = wrap.querySelectorAll('.stage');
  for (let i = 0; i < stageEls.length - 1; i++) {
    const fromNodes = Array.from(stageEls[i].querySelectorAll('.node'));
    const toNodes = Array.from(stageEls[i + 1].querySelectorAll('.node'));
    const fromTrainingOnly = stageEls[i].classList.contains('training-only');
    const toTrainingOnly = stageEls[i + 1].classList.contains('training-only');
    const dashed = fromTrainingOnly || toTrainingOnly;
    fromNodes.forEach(fn => {
      const fr = fn.getBoundingClientRect();
      const fx = fr.left + fr.width / 2 - wrapRect.left;
      const fy = fr.bottom - wrapRect.top;
      toNodes.forEach(tn => {
        const tr = tn.getBoundingClientRect();
        const tx = tr.left + tr.width / 2 - wrapRect.left;
        const ty = tr.top - wrapRect.top;
        const midY = (fy + ty) / 2;
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', 'M ' + fx + ',' + fy + ' C ' + fx + ',' + midY + ' ' + tx + ',' + midY + ' ' + tx + ',' + ty);
        path.setAttribute('stroke', '#9A9080');
        path.setAttribute('stroke-width', '1.4');
        path.setAttribute('fill', 'none');
        path.setAttribute('class', 'arrow-line');
        path.setAttribute('marker-end', 'url(#arrowhead)');
        if (dashed) path.setAttribute('stroke-dasharray', '5,4');
        svg.appendChild(path);
      });
    });
  }

  // Position edge labels (HTML divs) near their from-node bottom
  const labelGroups = wrap.querySelectorAll('.edge-labels');
  labelGroups.forEach(g => {
    Array.from(g.querySelectorAll('.edge-label')).forEach(lb => {
      const fromId = lb.dataset.from;
      const fromNode = wrap.querySelector('.node[data-id="' + fromId + '"]');
      if (!fromNode) return;
      const fr = fromNode.getBoundingClientRect();
      const cx = fr.left + fr.width / 2 - wrapRect.left;
      const by = fr.bottom - wrapRect.top;
      lb.style.left = cx + 'px';
      lb.style.top = (by + 8) + 'px';
    });
  });
}

function handleNodeClick(id) {
  const n = TREE[id];
  if (!n) return;
  if (n.type === 'composite') {
    state.path.push(id);
    render();
  } else {
    openDrawer(id);
  }
}

function openDrawer(id) {
  const n = TREE[id];
  const callCountBadge = n._callCount && n._callCount > 1
    ? '<span class="call-count">× ' + n._callCount + '</span>' : '';
  const functionalBadge = n.functional ? '<span class="badge functional">functional</span>' : '';
  const repeatBadge = n.repeat ? '<span class="badge">part of × ' + n.repeat + ' repeated block</span>' : '';

  const content = '<h2>' + n.name + (n.sub ? ' <span style="font-family: JetBrains Mono, monospace; font-size: 13px; color: var(--text-3); font-weight: 400;">· ' + n.sub + '</span>' : '') + '</h2>'
    + '<div class="qualname">' + (n._qualname || n.qualname || '') + callCountBadge + '</div>'
    + ((functionalBadge || repeatBadge) ? '<div class="badge-row">' + functionalBadge + repeatBadge + '</div>' : '')
    + '<h3>Shape</h3>'
    + '<div class="shape-box">' + (n._shape_html || n.shape_html || '—') + '</div>'
    + '<h3>What it does</h3>'
    + '<p>' + (n.what || '') + '</p>'
    + (n._act_chart_svg ? n._act_chart_svg : '')
    + (n.formula ? '<h3>Formula</h3><div class="formula">$$' + n.formula + '$$</div>' : '')
    + (n.callout ? '<div class="callout">' + n.callout + '</div>' : '')
    + '<h3>Source</h3>'
    + '<div class="src-link">' + (n._src_link
        ? '<a href="' + n._src_link + '" title="open in VSCode">' + (n._src || n.src || '—') + '</a>'
        : (n._src || n.src || '—')) + '</div>'
    + (n._trace_src && n._trace_src !== n._src
        ? '<div class="src-link src-link-trace">forward 实际执行处: '
          + (n._trace_src_link ? '<a href="' + n._trace_src_link + '">' + n._trace_src + '</a>' : n._trace_src)
          + '</div>'
        : '')
    + '<div class="drawer-actions">'
    +   '<button class="drawer-action-btn" onclick="copyModuleContext(\'' + id + '\')" title="把模块完整上下文复制成 markdown, 粘贴到 ChatGPT / Claude 网页用">'
    +     '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="4" y="3" width="9" height="11" rx="1"/><path d="M3 12V2h9"/></svg>'
    +     ' <span>复制上下文</span>'
    +   '</button>'
    +   '<button class="drawer-action-btn primary" onclick="window.AskAI && AskAI.openPanel({moduleId: \'' + id + '\'})" title="基于此模块向 AI 提问 (Ctrl+I 直接打开问答面板)">'
    +     '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 12 L13 4 M13 4 L9 4 M13 4 L13 8"/><circle cx="4" cy="13" r="1.2" fill="currentColor"/></svg>'
    +     ' <span>问 AI</span>'
    +   '</button>'
    + '</div>';
  document.getElementById('drawer-content').innerHTML = content;
  document.getElementById('drawer').classList.add('open');
  document.getElementById('drawer').setAttribute('aria-hidden', 'false');
  document.getElementById('backdrop').classList.add('open');
  state.drawerOpen = id;
  if (window.renderMathInElement) {
    renderMathInElement(document.getElementById('drawer-content'), {
      delimiters: [{left: '$$', right: '$$', display: true}, {left: '\\(', right: '\\)', display: false}]
    });
  }
}

function closeDrawer() {
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('drawer').setAttribute('aria-hidden', 'true');
  document.getElementById('backdrop').classList.remove('open');
  state.drawerOpen = null;
}

function render() {
  renderBreadcrumb();
  renderBlurb();
  renderCanvas();
}

// ---- Helpers for "Copy context" + Ask AI integration ----
function stripHtml(html) {
  if (!html) return '';
  const tmp = document.createElement('div');
  tmp.innerHTML = String(html).replace(/<br\s*\/?>/gi, '\n').replace(/&nbsp;/g, ' ');
  return (tmp.textContent || tmp.innerText || '').replace(/\s+\n/g, '\n').trim();
}

function buildModuleContextMarkdown(id) {
  const n = TREE[id];
  if (!n) return '';
  const breadcrumb = state.path.map(p => (TREE[p] && TREE[p].name) || p).join(' ▸ ');
  const lines = [];
  lines.push('# ' + (n.name || id) + (n.sub ? ' · ' + n.sub : ''));
  lines.push('');
  lines.push('**位置**: ' + breadcrumb);
  if (n._qualname || n.qualname) lines.push('**Qualname**: `' + (n._qualname || n.qualname) + '`');
  if (n._callCount && n._callCount > 1) lines.push('**调用次数**: × ' + n._callCount);
  if (n.functional) lines.push('**类型**: functional (非 nn.Module)');
  lines.push('**Shape**: ' + stripHtml(n._shape_html || n.shape_html || '—'));
  if (n.what) { lines.push(''); lines.push('## 说明'); lines.push(stripHtml(n.what)); }
  if (n.formula) { lines.push(''); lines.push('## 公式'); lines.push('$$' + n.formula + '$$'); }
  if (n.callout) { lines.push(''); lines.push('## 备注'); lines.push(stripHtml(n.callout)); }
  lines.push('');
  lines.push('**源码**: ' + (n._src || n.src || '—'));
  if (n._trace_src && n._trace_src !== n._src) {
    lines.push('**forward 实际执行处**: ' + n._trace_src);
  }
  return lines.join('\n');
}

function copyModuleContext(id) {
  const md = buildModuleContextMarkdown(id);
  navigator.clipboard.writeText(md).then(
    () => flashToast('已复制此模块的完整上下文 (可粘贴到 ChatGPT / Claude 网页)'),
    () => flashToast('复制失败 — 浏览器拒绝访问剪贴板', true)
  );
}

function flashToast(msg, isError) {
  let t = document.getElementById('toast');
  if (!t) {
    t = document.createElement('div');
    t.id = 'toast';
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.className = 'toast' + (isError ? ' error' : '') + ' show';
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.className = 'toast'; }, 2400);
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    // Ask AI panel 优先吃 Esc, 然后是 module drawer, 最后是 breadcrumb
    if (window.AskAI && AskAI.state.panelOpen) { AskAI.closePanel(); return; }
    if (state.drawerOpen) { closeDrawer(); return; }
    if (state.path.length > 1) { state.path.pop(); render(); }
  }
  // Ctrl/Cmd + I: 直接打开 Ask AI 面板
  if ((e.ctrlKey || e.metaKey) && (e.key === 'i' || e.key === 'I')) {
    e.preventDefault();
    if (window.AskAI) AskAI.openPanel({});
  }
});
document.getElementById('drawer-close').addEventListener('click', closeDrawer);
document.getElementById('backdrop').addEventListener('click', closeDrawer);

render();
"""


# ===========================================================================
# Ask AI widget — 嵌入式问答 (BYOK, OpenAI-compatible, 流式)
# ===========================================================================
ASK_AI_HTML = """
<!-- 浮动 Ask AI 按钮 (选中文字后出现) -->
<button id="ask-ai-fab" class="ask-ai-fab" hidden type="button" title="问 AI 关于这段内容 (Ctrl+I 也可)">
  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
    <path d="M3 12 L13 4"/><path d="M13 4 L9 4 M13 4 L13 8"/>
    <circle cx="4" cy="13" r="1.2" fill="currentColor"/>
  </svg>
  <span>问 AI</span>
</button>

<!-- Q&A 面板 -->
<aside id="ask-ai-panel" class="ask-ai-panel" hidden aria-hidden="true">
  <header class="ask-ai-panel-head">
    <div class="ask-ai-panel-title">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M3 12 L13 4"/><path d="M13 4 L9 4 M13 4 L13 8"/>
        <circle cx="4" cy="13" r="1.2" fill="currentColor"/>
      </svg>
      <span>问 AI</span>
    </div>
    <div class="ask-ai-panel-actions">
      <button class="ask-ai-icon-btn" id="ask-ai-reset" title="清空对话">
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 8a6 6 0 1 0 1.5-4M2 3v3h3"/></svg>
      </button>
      <button class="ask-ai-icon-btn" id="ask-ai-settings" title="API 设置">
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="2"/><path d="M8 1v2M8 13v2M1 8h2M13 8h2M3 3l1.5 1.5M11.5 11.5L13 13M3 13l1.5-1.5M11.5 4.5L13 3"/></svg>
      </button>
      <button class="ask-ai-icon-btn" id="ask-ai-close" title="关闭 (Esc)">×</button>
    </div>
  </header>
  <div class="ask-ai-context" id="ask-ai-context-pill" hidden></div>
  <div class="ask-ai-messages" id="ask-ai-messages">
    <div class="ask-ai-empty">
      <p>选中页面上任何一段文字 + 点浮动按钮，或者直接在下方输入问题。</p>
      <p class="ask-ai-empty-sub">AI 会自动收到你当前所在的 breadcrumb 路径和打开的模块详情, 不需要你重复描述。</p>
    </div>
  </div>
  <form class="ask-ai-input-row" id="ask-ai-form">
    <textarea id="ask-ai-input" class="ask-ai-input" rows="2" placeholder="问点什么... (Ctrl+Enter 发送)" autocomplete="off"></textarea>
    <button type="submit" class="ask-ai-send" id="ask-ai-send" title="发送 (Ctrl+Enter)">
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 8 L14 2 L11 14 L8 9 L2 8 Z"/></svg>
    </button>
  </form>
  <div class="ask-ai-footer">
    <div class="ask-ai-model-switcher" id="model-switcher">
      <button type="button" class="ask-ai-model-tag" id="ask-ai-model-tag" title="点击切换 model">未配置</button>
      <div class="ask-model-dropdown" id="model-dropdown" hidden></div>
    </div>
    <span class="ask-ai-hint">Ctrl+I 唤起 · Esc 关闭</span>
  </div>
</aside>

<!-- BYOK 配置模态框 -->
<div id="ask-ai-config-backdrop" class="ask-ai-config-backdrop" hidden></div>
<div id="ask-ai-config" class="ask-ai-config" hidden role="dialog" aria-modal="true">
  <header class="ask-ai-config-head">
    <h3>API 配置 (BYOK)</h3>
    <button class="ask-ai-icon-btn" id="ask-ai-config-close">×</button>
  </header>
  <div class="ask-ai-config-body">
    <div class="ask-ai-warn">
      <strong>注意:</strong> API key 仅保存在你浏览器本地 (localStorage)，不会上传任何服务器。HTML 是离线运行的，所有 API 请求直接从你的浏览器发到你配置的端点。<br>
      <strong>不要在以下场景使用:</strong> 把 HTML 部署到公开网站 / 多人共享这台机器 / 通过不可信网络访问。
    </div>

    <label class="ask-ai-field">
      <span>Provider</span>
      <select id="cfg-provider">
        <option value="gemini">Gemini (OpenAI-compatible)</option>
        <option value="openai">OpenAI</option>
        <option value="deepseek">DeepSeek</option>
        <option value="ollama">Ollama (本地)</option>
        <option value="custom">自定义</option>
      </select>
    </label>

    <div class="ask-ai-section-title">Provider 配置</div>
    <label class="ask-ai-field">
      <span>API Base URL</span>
      <input id="cfg-base-url" type="text" placeholder="https://...">
    </label>
    <label class="ask-ai-field">
      <span>API Key</span>
      <input id="cfg-api-key" type="password" placeholder="sk-... / Ollama 本地可留空">
    </label>
    <div class="ask-ai-field">
      <span>Model</span>
      <div class="ask-ai-model-row">
        <select id="cfg-model-select" class="ask-ai-model-select" hidden></select>
        <input id="cfg-model-input" type="text" class="ask-ai-model-input" placeholder="例: gemini-2.5-flash">
        <button id="cfg-model-refresh" type="button" class="ask-ai-icon-btn-inline" title="从 /v1/models 拉取可用 model 列表">
          <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M2 8a6 6 0 1 0 1.5-4M2 3v3h3"/>
          </svg>
        </button>
      </div>
      <div id="cfg-model-hint" class="ask-ai-field-hint">↻ 点刷新可拉取 model 列表; 拉取失败时保持手填</div>
    </div>

    <div class="ask-ai-section-title">通用 inference 参数</div>
    <div class="ask-ai-field-grid">
      <label class="ask-ai-field">
        <span>Temperature</span>
        <input id="cfg-temp" type="number" min="0" max="2" step="0.05" value="0.4">
      </label>
      <label class="ask-ai-field">
        <span>Max tokens</span>
        <input id="cfg-max-tokens" type="number" min="64" step="64" value="8192">
      </label>
      <label class="ask-ai-field">
        <span>Top-p</span>
        <input id="cfg-top-p" type="number" min="0" max="1" step="0.05" value="1.0">
      </label>
      <label class="ask-ai-field">
        <span>对话保留轮数</span>
        <select id="cfg-retention">
          <option value="1">1</option>
          <option value="3">3</option>
          <option value="5" selected>5</option>
          <option value="10">10</option>
          <option value="-1">无限</option>
        </select>
      </label>
    </div>
    <label class="ask-ai-field">
      <span>System prompt (留空用默认)</span>
      <textarea id="cfg-sys-prompt" rows="3" placeholder="留空 = 用内置默认 prompt (强调 markdown 格式 + 简洁 + 用户当前位置感知)"></textarea>
    </label>

    <div class="ask-ai-config-row">
      <button class="ask-ai-btn" id="cfg-test" type="button">测试连接</button>
      <button class="ask-ai-btn primary" id="cfg-save" type="button">保存</button>
    </div>
    <div class="ask-ai-config-status" id="cfg-status"></div>

    <details class="ask-ai-help">
      <summary>常见问题</summary>
      <p><strong>CORS 错误?</strong> 如果用 <code>file://</code> 直接打开 HTML 撞 CORS, 试试用本地 server: <code>cd outputs/traces/v18_demo &amp;&amp; python -m http.server 8000</code>, 然后访问 <code>http://localhost:8000/latest.html</code>。</p>
      <p><strong>Gemini key?</strong> 在 <a href="https://aistudio.google.com/apikey" target="_blank" rel="noopener">aistudio.google.com/apikey</a> 申请。免费额度够本工具使用。</p>
      <p><strong>Ollama 本地?</strong> 启动 <code>ollama serve</code> 后, base URL 填 <code>http://localhost:11434/v1</code>, key 留空, model 点 ↻ 自动拉到你 pull 过的。</p>
      <p><strong>Model 拉不到?</strong> 有些 endpoint 不支持 <code>/v1/models</code> 列表 (尤其 Gemini OpenAI-compat 层)。保持手填模式即可, 不影响 chat 使用。</p>
    </details>
  </div>
</div>
"""
# 提示 toast (复用 module drawer 的 Copy 上下文 按钮)
ASK_AI_HTML += """
<div id="toast" class="toast"></div>
"""


ASK_AI_CSS = r"""
/* ============= Ask AI widget ============= */
.ask-ai-fab {
  position: absolute;
  z-index: 200;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 6px 10px 6px 8px;
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 12px;
  font-weight: 500;
  color: white;
  background: var(--accent);
  border: none;
  border-radius: 6px;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(199, 71, 42, 0.32), 0 1px 2px rgba(0,0,0,0.08);
  transition: transform 0.12s ease, opacity 0.12s ease, background 0.12s;
  transform-origin: bottom left;
  animation: ask-fab-in 0.12s ease-out;
}
.ask-ai-fab:hover { background: var(--accent-2); }
.ask-ai-fab:active { transform: translateY(0.5px); }
@keyframes ask-fab-in {
  from { opacity: 0; transform: scale(0.9); }
  to   { opacity: 1; transform: scale(1); }
}

.ask-ai-panel {
  position: fixed;
  top: 0; right: 0; bottom: 0;
  width: 420px;
  max-width: 96vw;
  background: var(--bg);
  border-left: 0.5px solid var(--border);
  z-index: 110;
  display: flex;
  flex-direction: column;
  transform: translateX(100%);
  transition: transform 0.24s cubic-bezier(0.2, 0.8, 0.2, 1);
  box-shadow: -8px 0 32px rgba(50, 30, 10, 0.10);
}
.ask-ai-panel.open { transform: translateX(0); }
/* 屏幕窄时让 Ask AI 面板覆盖在 module drawer 之上 */
@media (max-width: 1200px) {
  .ask-ai-panel { width: 100vw; }
  .drawer.open ~ .ask-ai-panel.open { z-index: 120; }
}

.ask-ai-panel-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 16px 12px;
  border-bottom: 0.5px solid var(--border);
}
.ask-ai-panel-title {
  display: inline-flex; align-items: center; gap: 8px;
  font-family: 'Bricolage Grotesque', serif;
  font-weight: 600; font-size: 15px;
  color: var(--accent-2);
}
.ask-ai-panel-actions { display: flex; gap: 4px; }

.ask-ai-icon-btn {
  width: 26px; height: 26px;
  display: inline-flex; align-items: center; justify-content: center;
  background: transparent; border: none; cursor: pointer;
  color: var(--text-2); border-radius: 5px;
  font-size: 16px; line-height: 1;
  transition: background 0.12s, color 0.12s;
}
.ask-ai-icon-btn:hover { background: var(--surface-alt); color: var(--text); }

.ask-ai-context {
  margin: 10px 16px 0;
  padding: 8px 12px;
  background: var(--accent-soft-2);
  border-left: 3px solid var(--accent);
  border-radius: 0 5px 5px 0;
  font-size: 11.5px;
  color: var(--text-2);
  line-height: 1.55;
  font-family: 'JetBrains Mono', monospace;
}
.ask-ai-context .ctx-breadcrumb { color: var(--accent-2); font-weight: 500; }
.ask-ai-context .ctx-sel { color: var(--text); font-style: italic; }

.ask-ai-messages {
  flex: 1;
  overflow-y: auto;
  padding: 14px 16px;
  display: flex; flex-direction: column; gap: 12px;
  font-size: 13.5px; line-height: 1.65;
}
.ask-ai-empty {
  text-align: center; color: var(--text-3);
  padding: 28px 8px 10px;
  font-style: italic;
}
.ask-ai-empty p { margin: 0 0 8px; font-size: 13px; }
.ask-ai-empty-sub { font-size: 11.5px; color: var(--text-3); }

.ask-msg { display: flex; gap: 8px; }
.ask-msg-role {
  flex-shrink: 0; width: 26px; height: 26px;
  border-radius: 5px;
  display: inline-flex; align-items: center; justify-content: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; font-weight: 600;
  letter-spacing: 0.5px;
}
.ask-msg.user .ask-msg-role { background: var(--accent); color: white; }
.ask-msg.assistant .ask-msg-role { background: var(--surface-alt); color: var(--text-2); }
.ask-msg-body {
  flex: 1; min-width: 0;
  background: var(--surface);
  border: 0.5px solid var(--border);
  border-radius: 6px;
  padding: 9px 12px;
  white-space: pre-wrap; word-break: break-word;
}
.ask-msg.user .ask-msg-body {
  background: var(--accent-soft-2);
  border-color: var(--accent-soft);
}
.ask-msg.assistant.streaming .ask-msg-body::after {
  content: '▎'; color: var(--accent); animation: ask-blink 0.9s steps(2) infinite;
}
@keyframes ask-blink { 50% { opacity: 0.2; } }
.ask-msg-body code {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px; background: var(--surface-alt);
  padding: 1px 4px; border-radius: 3px;
  border: 0.5px solid var(--border);
}
.ask-msg-body pre {
  background: var(--code-bg);
  color: var(--code-ink);
  padding: 8px 10px; border-radius: 4px;
  overflow-x: auto; font-size: 11.5px;
  margin: 6px 0;
}

.ask-error {
  background: #FFE9E0;
  border-left: 3px solid #C04020;
  color: #7A2010;
  padding: 8px 12px;
  border-radius: 0 5px 5px 0;
  font-size: 12.5px;
  margin: 6px 0;
}
.ask-error button {
  margin-left: 8px;
  font-size: 11px;
  padding: 2px 8px;
  background: white;
  border: 0.5px solid #C04020;
  color: #7A2010;
  border-radius: 3px; cursor: pointer;
}

.ask-ai-input-row {
  padding: 10px 12px 8px;
  border-top: 0.5px solid var(--border);
  display: flex; gap: 8px; align-items: flex-end;
}
.ask-ai-input {
  flex: 1;
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 13px; line-height: 1.5;
  border: 0.5px solid var(--border-strong);
  border-radius: 6px;
  padding: 8px 10px;
  background: var(--surface);
  color: var(--text);
  resize: none;
  max-height: 200px;
  outline: none;
}
.ask-ai-input:focus { border-color: var(--accent); }
.ask-ai-send {
  width: 32px; height: 32px;
  background: var(--accent);
  color: white;
  border: none; border-radius: 6px;
  cursor: pointer;
  display: inline-flex; align-items: center; justify-content: center;
  transition: background 0.12s;
}
.ask-ai-send:hover { background: var(--accent-2); }
.ask-ai-send:disabled { background: var(--text-3); cursor: not-allowed; }
.ask-ai-send.stop-mode {
  background: #7A2010;
  animation: ask-stop-pulse 1.4s ease-in-out infinite;
}
.ask-ai-send.stop-mode:hover { background: #5A1808; }
@keyframes ask-stop-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(122, 32, 16, 0.4); }
  50%      { box-shadow: 0 0 0 4px rgba(122, 32, 16, 0); }
}

.ask-ai-footer {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 14px 10px;
  font-size: 10.5px;
  color: var(--text-3);
  font-family: 'JetBrains Mono', monospace;
}
.ask-ai-model-tag {
  background: var(--surface-alt);
  padding: 2px 8px;
  border-radius: 3px;
  border: 0.5px solid var(--border);
}
.ask-ai-model-tag.unconfigured { color: var(--accent-2); border-color: var(--accent-soft); background: var(--accent-soft-2); }

/* ----- BYOK config modal ----- */
.ask-ai-config-backdrop {
  position: fixed; inset: 0;
  background: rgba(31, 27, 22, 0.42);
  z-index: 200;
}
.ask-ai-config {
  position: fixed;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  width: 440px; max-width: 92vw;
  max-height: 88vh; overflow-y: auto;
  background: var(--bg);
  border: 0.5px solid var(--border);
  border-radius: 10px;
  z-index: 201;
  box-shadow: 0 20px 60px rgba(50, 30, 10, 0.18);
}
.ask-ai-config-head {
  display: flex; justify-content: space-between; align-items: center;
  padding: 14px 18px 12px;
  border-bottom: 0.5px solid var(--border);
}
.ask-ai-config-head h3 {
  font-family: 'Bricolage Grotesque', serif;
  font-weight: 600; font-size: 16px; margin: 0;
}
.ask-ai-config-body { padding: 14px 18px 18px; }
.ask-ai-warn {
  background: #FFF6E5;
  border-left: 3px solid #D49620;
  padding: 9px 12px;
  font-size: 12px;
  color: #6A4A10;
  border-radius: 0 5px 5px 0;
  margin-bottom: 14px;
  line-height: 1.55;
}
.ask-ai-field { display: block; margin: 10px 0; }
.ask-ai-field > span {
  display: block;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-2);
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.ask-ai-field input, .ask-ai-field select {
  width: 100%;
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 13px;
  padding: 7px 10px;
  border: 0.5px solid var(--border-strong);
  border-radius: 5px;
  background: var(--surface);
  color: var(--text);
  outline: none;
}
.ask-ai-field input:focus, .ask-ai-field select:focus { border-color: var(--accent); }
.ask-ai-config-row {
  display: flex; gap: 10px; margin-top: 14px;
}
.ask-ai-btn {
  padding: 7px 14px;
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 13px;
  background: var(--surface);
  border: 0.5px solid var(--border-strong);
  border-radius: 5px;
  cursor: pointer;
  color: var(--text);
  transition: background 0.12s;
}
.ask-ai-btn:hover { background: var(--surface-alt); }
.ask-ai-btn.primary { background: var(--accent); color: white; border-color: var(--accent); }
.ask-ai-btn.primary:hover { background: var(--accent-2); border-color: var(--accent-2); }
.ask-ai-config-status {
  margin-top: 10px;
  font-size: 12px;
  font-family: 'JetBrains Mono', monospace;
  min-height: 16px;
}
.ask-ai-config-status.ok { color: #2D7A45; }
.ask-ai-config-status.err { color: #C04020; }
.ask-ai-help {
  margin-top: 14px;
  font-size: 11.5px;
  color: var(--text-2);
}
.ask-ai-help summary { cursor: pointer; color: var(--text-2); margin-bottom: 6px; }
.ask-ai-help p { margin: 6px 0; line-height: 1.55; }
.ask-ai-help code {
  font-family: 'JetBrains Mono', monospace;
  background: var(--surface-alt);
  padding: 1px 5px; border-radius: 3px;
  font-size: 11px;
}

/* 配置 modal 新增元素 */
.ask-ai-section-title {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent-2);
  margin: 16px 0 4px;
  padding-bottom: 4px;
  border-bottom: 0.5px solid var(--accent-soft);
}
.ask-ai-model-row {
  display: flex; gap: 6px; align-items: stretch;
}
.ask-ai-model-row select, .ask-ai-model-row input {
  flex: 1;
}
.ask-ai-icon-btn-inline {
  flex-shrink: 0;
  width: 32px;
  display: inline-flex; align-items: center; justify-content: center;
  background: var(--surface);
  border: 0.5px solid var(--border-strong);
  border-radius: 5px;
  cursor: pointer;
  color: var(--text-2);
  transition: background 0.12s, color 0.12s;
}
.ask-ai-icon-btn-inline:hover { background: var(--surface-alt); color: var(--accent-2); border-color: var(--accent-soft); }
.ask-ai-icon-btn-inline.loading svg { animation: ask-refresh-spin 0.8s linear infinite; }
@keyframes ask-refresh-spin { from { transform: rotate(0); } to { transform: rotate(360deg); } }
.ask-ai-field-hint {
  font-size: 10.5px;
  color: var(--text-3);
  margin-top: 4px;
  font-family: 'JetBrains Mono', monospace;
}
.ask-ai-field-hint.ok  { color: #2D7A45; }
.ask-ai-field-hint.err { color: #C04020; }
.ask-ai-field-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px 12px;
}
.ask-ai-field textarea {
  width: 100%;
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 12.5px;
  padding: 7px 10px;
  border: 0.5px solid var(--border-strong);
  border-radius: 5px;
  background: var(--surface);
  color: var(--text);
  outline: none;
  resize: vertical;
  min-height: 60px;
}
.ask-ai-field textarea:focus { border-color: var(--accent); }

/* In-panel model quickswitch (footer 里点 model tag 弹小菜单) */
.ask-ai-model-switcher { position: relative; }
.ask-ai-model-tag { cursor: pointer; user-select: none; transition: background 0.12s; }
.ask-ai-model-tag:hover { background: var(--accent-soft-2); border-color: var(--accent-soft); color: var(--accent-2); }
.ask-ai-model-tag::after { content: ' ⌄'; opacity: 0.6; }
.ask-model-dropdown {
  position: absolute;
  bottom: calc(100% + 4px);
  left: 0;
  min-width: 180px;
  max-height: 240px;
  overflow-y: auto;
  background: var(--surface);
  border: 0.5px solid var(--border-strong);
  border-radius: 6px;
  box-shadow: 0 6px 24px rgba(50, 30, 10, 0.16);
  z-index: 130;
  padding: 4px 0;
}
.ask-model-dropdown button {
  display: block;
  width: 100%;
  text-align: left;
  background: none;
  border: none;
  padding: 6px 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11.5px;
  color: var(--text);
  cursor: pointer;
}
.ask-model-dropdown button:hover { background: var(--surface-alt); color: var(--accent-2); }
.ask-model-dropdown button.current { background: var(--accent-soft-2); color: var(--accent-2); font-weight: 500; }
.ask-model-dropdown .ask-model-dropdown-empty {
  padding: 8px 12px;
  font-size: 11px;
  color: var(--text-3);
  font-style: italic;
}

/* 切模型的清空对话提示 */
.ask-switch-confirm {
  position: absolute;
  bottom: calc(100% + 4px);
  left: 0;
  width: 260px;
  background: var(--surface);
  border: 0.5px solid var(--accent-soft);
  border-left: 3px solid var(--accent);
  border-radius: 5px;
  padding: 10px 12px;
  font-size: 12px;
  line-height: 1.55;
  z-index: 131;
  box-shadow: 0 6px 24px rgba(50, 30, 10, 0.16);
}
.ask-switch-confirm-actions {
  display: flex; gap: 6px; margin-top: 8px;
}
.ask-switch-confirm-actions button {
  flex: 1;
  font-size: 11px;
  padding: 5px 8px;
  border-radius: 4px;
  border: 0.5px solid var(--border-strong);
  background: var(--surface);
  cursor: pointer;
  font-family: 'DM Sans', system-ui, sans-serif;
}
.ask-switch-confirm-actions button.primary {
  background: var(--accent); color: white; border-color: var(--accent);
}
.ask-switch-confirm-actions button:hover { background: var(--surface-alt); }
.ask-switch-confirm-actions button.primary:hover { background: var(--accent-2); }

/* Thinking section in assistant messages */
.ask-thinking {
  margin-bottom: 8px;
  font-size: 12px;
  color: var(--text-2);
}
.ask-thinking > summary {
  cursor: pointer;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  color: var(--text-3);
  padding: 3px 8px;
  background: var(--surface-alt);
  border-radius: 4px;
  display: inline-block;
  user-select: none;
}
.ask-thinking > summary::before {
  content: '🧠 ';
  margin-right: 2px;
}
.ask-thinking[open] > summary { color: var(--accent-2); }
.ask-thinking-content {
  margin: 6px 0 4px;
  padding: 8px 12px;
  background: rgba(0,0,0,0.02);
  border-left: 2px solid var(--border-strong);
  font-style: italic;
  font-size: 12px;
  color: var(--text-2);
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}

/* Markdown 渲染质量 — 紧凑行距, 段落间距小, 标题间距明确但不夸张 */
.ask-msg-body { line-height: 1.55; }
.ask-msg-body p { margin: 0.25em 0; }
.ask-msg-body p:first-child { margin-top: 0; }
.ask-msg-body p:last-child { margin-bottom: 0; }
.ask-msg-body p:empty { display: none; }
.ask-msg-body ul, .ask-msg-body ol { margin: 0.3em 0; padding-left: 1.4em; }
.ask-msg-body li { margin-bottom: 0.2em; line-height: 1.55; }
.ask-msg-body li:last-child { margin-bottom: 0; }
.ask-msg-body h1, .ask-msg-body h2, .ask-msg-body h3, .ask-msg-body h4 {
  font-family: 'Bricolage Grotesque', serif;
  font-weight: 600;
  margin: 0.7em 0 0.25em;
  line-height: 1.25;
  color: var(--text);
}
.ask-msg-body h1:first-child, .ask-msg-body h2:first-child,
.ask-msg-body h3:first-child, .ask-msg-body h4:first-child { margin-top: 0; }
.ask-msg-body h1 { font-size: 15px; }
.ask-msg-body h2 { font-size: 14px; }
.ask-msg-body h3 { font-size: 13.5px; color: var(--accent-2); }
.ask-msg-body h4 { font-size: 13px; }
.ask-msg-body strong { color: var(--accent-2); font-weight: 600; }
.ask-msg-body hr { border: none; border-top: 0.5px solid var(--border); margin: 0.6em 0; }
.ask-msg-body blockquote {
  border-left: 3px solid var(--accent-soft);
  margin: 0.4em 0;
  padding: 0.1em 0 0.1em 10px;
  color: var(--text-2);
}
.ask-msg-body pre { margin: 0.4em 0; }
/* thinking 跟 content 之间留 minimal gap */
.ask-thinking { margin-bottom: 4px; }
.ask-thinking-content { margin: 4px 0 0; padding: 6px 10px; line-height: 1.5; }
.ask-content { margin-top: 0; }
.ask-content > *:first-child { margin-top: 0; }

/* drawer 底部操作按钮 */
.drawer-actions {
  display: flex; gap: 8px;
  margin-top: 22px;
  padding-top: 16px;
  border-top: 0.5px dashed var(--border);
}
.drawer-action-btn {
  flex: 1;
  display: inline-flex; align-items: center; justify-content: center; gap: 6px;
  padding: 8px 10px;
  font-family: 'DM Sans', system-ui, sans-serif;
  font-size: 12.5px;
  color: var(--text);
  background: var(--surface);
  border: 0.5px solid var(--border-strong);
  border-radius: 5px;
  cursor: pointer;
  transition: background 0.12s, color 0.12s, border-color 0.12s;
}
.drawer-action-btn:hover { background: var(--surface-alt); border-color: var(--text-2); }
.drawer-action-btn.primary {
  background: var(--accent);
  color: white;
  border-color: var(--accent);
}
.drawer-action-btn.primary:hover { background: var(--accent-2); border-color: var(--accent-2); }

/* Toast */
.toast {
  position: fixed;
  left: 50%; bottom: 32px;
  transform: translateX(-50%) translateY(20px);
  background: var(--text);
  color: #FAF7F0;
  padding: 9px 16px;
  border-radius: 6px;
  font-size: 12.5px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.18);
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.18s, transform 0.18s;
  z-index: 250;
}
.toast.show {
  opacity: 1;
  transform: translateX(-50%) translateY(0);
}
.toast.error { background: #7A2010; }
"""


ASK_AI_JS = r"""
// =========== Ask AI widget — BYOK, OpenAI-compatible, streaming ============
window.AskAI = (function() {
  const PROVIDER_DEFAULTS = {
    gemini:   { base: 'https://generativelanguage.googleapis.com/v1beta/openai', model: 'gemini-2.5-flash' },
    openai:   { base: 'https://api.openai.com/v1', model: 'gpt-4o-mini' },
    deepseek: { base: 'https://api.deepseek.com/v1', model: 'deepseek-chat' },
    ollama:   { base: 'http://localhost:11434/v1', model: 'llama3.2' },
    custom:   { base: '', model: '' },
  };
  const CONFIG_KEY = 'askAI.config';

  const S = {
    panelOpen: false,
    configOpen: false,
    selection: null,
    messages: [],
    config: null,
    isStreaming: false,
    abortCtrl: null,
    lastSentContextHash: '',  // compact context: 跟踪上次已发送的 context, 后续轮只在变化时重发
  };

  // ---- Config storage ----
  function loadConfig() {
    try {
      const raw = localStorage.getItem(CONFIG_KEY);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch (e) { return null; }
  }
  function saveConfig(cfg) {
    localStorage.setItem(CONFIG_KEY, JSON.stringify(cfg));
    S.config = cfg;
    updateModelTag();
  }
  function updateModelTag() {
    const tag = document.getElementById('ask-ai-model-tag');
    if (!tag) return;
    if (S.config && S.config.model) {
      tag.textContent = S.config.model;
      tag.classList.remove('unconfigured');
    } else {
      tag.textContent = '未配置 (点 ⚙ 设置)';
      tag.classList.add('unconfigured');
    }
  }

  // ---- Selection floating button ----
  // 统一入口 recheckSelection: selectionchange / mouseup / keyup 都进这里。
  // 用 debounce 避免 drag 期间 selectionchange 高频触发 (浏览器在 drag 中每像素 fire 一次)。
  // 关键: 不依赖 mouseup 单点 — 之前的 bug 就是 mouseup 在某些场景不触发或被吃掉。
  let _selTimer = null;
  function recheckSelection() {
    if (_selTimer) clearTimeout(_selTimer);
    _selTimer = setTimeout(_applySelection, 100);
  }
  function _applySelection() {
    const sel = window.getSelection();
    const fab = document.getElementById('ask-ai-fab');
    if (!sel || sel.rangeCount === 0 || sel.isCollapsed) {
      fab.hidden = true; S.selection = null; return;
    }
    const text = sel.toString().trim();
    if (text.length < 5) { fab.hidden = true; S.selection = null; return; }
    // 选区在 Ask AI 自己的 panel/config 里 → 不弹 FAB (避免重叠)
    const anchor = sel.anchorNode;
    const anchorEl = anchor && (anchor.nodeType === 1 ? anchor : anchor.parentElement);
    if (anchorEl && anchorEl.closest && anchorEl.closest('.ask-ai-panel, .ask-ai-config, #ask-ai-fab, #toast')) {
      fab.hidden = true; return;
    }
    // 取选区最后一行 rect (跨多行时按钮才贴近视线终点)
    const range = sel.getRangeAt(0);
    const rects = range.getClientRects();
    if (!rects.length) { fab.hidden = true; return; }
    const lastRect = rects[rects.length - 1];
    if (lastRect.width === 0 && lastRect.height === 0) { fab.hidden = true; return; }
    S.selection = text;
    fab.style.top = (window.scrollY + lastRect.bottom + 6) + 'px';
    fab.style.left = (window.scrollX + lastRect.right + 4) + 'px';
    fab.hidden = false;
  }
  function hideFab() {
    document.getElementById('ask-ai-fab').hidden = true;
  }

  // ---- Panel ----
  function openPanel(opts) {
    opts = opts || {};
    if (!S.config || !S.config.apiKey && S.config.provider !== 'ollama') {
      // First-time: open config first (but allow ollama with empty key)
      if (!S.config) { openConfig(true); return; }
    }
    S.panelOpen = true;
    const panel = document.getElementById('ask-ai-panel');
    panel.classList.add('open');
    panel.hidden = false;
    panel.setAttribute('aria-hidden', 'false');
    renderContextPill(opts);
    setTimeout(() => document.getElementById('ask-ai-input').focus(), 240);
  }
  function closePanel() {
    S.panelOpen = false;
    const panel = document.getElementById('ask-ai-panel');
    panel.classList.remove('open');
    panel.setAttribute('aria-hidden', 'true');
    if (S.abortCtrl) { S.abortCtrl.abort(); S.abortCtrl = null; }
    setTimeout(() => { panel.hidden = true; }, 260);
  }
  function resetConversation() {
    S.messages = [];
    S.lastSentContextHash = '';  // 重置 context cache, 下次发会重新带完整 context
    const msgs = document.getElementById('ask-ai-messages');
    msgs.innerHTML = '<div class="ask-ai-empty">'
      + '<p>选中页面上任何一段文字 + 点浮动按钮，或者直接在下方输入问题。</p>'
      + '<p class="ask-ai-empty-sub">AI 会自动收到你当前所在的 breadcrumb 路径和打开的模块详情, 不需要你重复描述。</p>'
      + '</div>';
  }

  function renderContextPill(opts) {
    const pill = document.getElementById('ask-ai-context-pill');
    const breadcrumb = state.path.map(p => (TREE[p] && TREE[p].name) || p).join(' ▸ ');
    let html = '<div><span class="ctx-breadcrumb">' + breadcrumb + '</span></div>';
    const moduleId = opts.moduleId || state.drawerOpen;
    if (moduleId && TREE[moduleId]) {
      html += '<div>聚焦模块: <span class="ctx-breadcrumb">' + (TREE[moduleId].name || moduleId) + '</span></div>';
    }
    if (S.selection) {
      const truncated = S.selection.length > 80 ? S.selection.slice(0, 80) + '…' : S.selection;
      html += '<div>选中: <span class="ctx-sel">"' + escapeHtml(truncated) + '"</span></div>';
    }
    if (breadcrumb || moduleId || S.selection) {
      pill.innerHTML = html;
      pill.hidden = false;
    } else {
      pill.hidden = true;
    }
  }

  // ---- Config modal ----
  function openConfig(firstTime) {
    S.configOpen = true;
    document.getElementById('ask-ai-config-backdrop').hidden = false;
    const modal = document.getElementById('ask-ai-config');
    modal.hidden = false;
    // populate
    const cfg = S.config || {};
    const defaults = PROVIDER_DEFAULTS[cfg.provider || 'gemini'] || PROVIDER_DEFAULTS.gemini;
    document.getElementById('cfg-provider').value = cfg.provider || 'gemini';
    document.getElementById('cfg-base-url').value = cfg.baseUrl || defaults.base;
    document.getElementById('cfg-api-key').value = cfg.apiKey || '';
    // model: 优先用缓存的 list 填 select, 否则手填 input
    const cached = cfg.cachedModels || [];
    const sel = document.getElementById('cfg-model-select');
    const inp = document.getElementById('cfg-model-input');
    if (cached.length) {
      sel.innerHTML = cached.map(m =>
        '<option value="' + escapeHtml(m) + '">' + escapeHtml(m) + '</option>'
      ).join('');
      if (cfg.model && cached.includes(cfg.model)) sel.value = cfg.model;
      sel.hidden = false; inp.hidden = true;
      document.getElementById('cfg-model-hint').textContent = '使用上次拉到的 ' + cached.length + ' 个 model (↻ 重新拉取)';
      document.getElementById('cfg-model-hint').className = 'ask-ai-field-hint ok';
    } else {
      sel.hidden = true; inp.hidden = false;
      inp.value = cfg.model || defaults.model;
      document.getElementById('cfg-model-hint').textContent = '↻ 点刷新可从 /v1/models 拉取列表';
      document.getElementById('cfg-model-hint').className = 'ask-ai-field-hint';
    }
    // inference 参数
    document.getElementById('cfg-temp').value      = cfg.temperature != null ? cfg.temperature : 0.4;
    document.getElementById('cfg-max-tokens').value = cfg.maxTokens != null ? cfg.maxTokens : 8192;
    document.getElementById('cfg-top-p').value     = cfg.topP != null ? cfg.topP : 1.0;
    document.getElementById('cfg-retention').value = cfg.retention != null ? cfg.retention : 5;
    document.getElementById('cfg-sys-prompt').value = cfg.systemPrompt || '';

    document.getElementById('cfg-status').textContent = firstTime ? '👋 第一次使用 — 选 provider 后保存即可 (高级参数留默认就行)' : '';
    document.getElementById('cfg-status').className = 'ask-ai-config-status';
  }
  function closeConfig() {
    S.configOpen = false;
    document.getElementById('ask-ai-config-backdrop').hidden = true;
    document.getElementById('ask-ai-config').hidden = true;
  }
  function applyProviderPreset(provider) {
    const preset = PROVIDER_DEFAULTS[provider] || PROVIDER_DEFAULTS.custom;
    document.getElementById('cfg-base-url').value = preset.base;
    document.getElementById('cfg-model-input').value = preset.model;
    // 切 provider 时 model 列表失效, 回到手填
    document.getElementById('cfg-model-select').hidden = true;
    document.getElementById('cfg-model-input').hidden = false;
    document.getElementById('cfg-model-hint').textContent = '↻ 点刷新可从该 provider 拉取 model 列表';
    document.getElementById('cfg-model-hint').className = 'ask-ai-field-hint';
  }

  function collectConfig() {
    const sel = document.getElementById('cfg-model-select');
    const inp = document.getElementById('cfg-model-input');
    const modelFromSelect = !sel.hidden && sel.value;
    const modelFromInput = inp.value.trim();
    return {
      provider: document.getElementById('cfg-provider').value,
      baseUrl:  document.getElementById('cfg-base-url').value.trim(),
      apiKey:   document.getElementById('cfg-api-key').value.trim(),
      model:    modelFromSelect || modelFromInput,
      temperature: parseFloat(document.getElementById('cfg-temp').value) || 0.4,
      maxTokens:   parseInt(document.getElementById('cfg-max-tokens').value, 10) || 8192,
      topP:        parseFloat(document.getElementById('cfg-top-p').value) || 1.0,
      retention:   parseInt(document.getElementById('cfg-retention').value, 10),
      systemPrompt: document.getElementById('cfg-sys-prompt').value,
      // 缓存最近一次拉到的 model list, 给 in-panel 快切器用
      cachedModels: S.config && S.config.cachedModels || [],
    };
  }

  // ---- /v1/models 拉取 ----
  async function fetchModelList() {
    const cfg = collectConfig();
    if (!cfg.baseUrl) return { error: '先填 Base URL' };
    const url = cfg.baseUrl.replace(/\/+$/, '') + '/models';
    try {
      const headers = {};
      if (cfg.apiKey) headers['Authorization'] = 'Bearer ' + cfg.apiKey;
      const resp = await fetch(url, { headers });
      if (!resp.ok) {
        if (resp.status === 401) return { error: '401 (API key 无效)' };
        if (resp.status === 403) return { error: '403 (拒绝访问)' };
        if (resp.status === 404) return { error: '404 (endpoint 不支持 /models)' };
        return { error: 'HTTP ' + resp.status };
      }
      const data = await resp.json();
      let models = [];
      if (Array.isArray(data)) models = data.map(m => typeof m === 'string' ? m : (m.id || m.name));
      else if (data.data && Array.isArray(data.data)) models = data.data.map(m => m.id || m.name);
      else if (data.models && Array.isArray(data.models)) models = data.models.map(m => typeof m === 'string' ? m : (m.id || m.name || m.model));
      models = models.filter(Boolean);
      if (!models.length) return { error: 'endpoint 返回空列表' };
      return { models };
    } catch (e) {
      const msg = e.message || String(e);
      const cors = msg.includes('Failed to fetch') || msg.includes('NetworkError') ? ' (可能 CORS)' : '';
      return { error: msg + cors };
    }
  }

  async function refreshModels() {
    const btn = document.getElementById('cfg-model-refresh');
    const hint = document.getElementById('cfg-model-hint');
    const sel = document.getElementById('cfg-model-select');
    const inp = document.getElementById('cfg-model-input');
    btn.classList.add('loading');
    hint.textContent = '拉取中...';
    hint.className = 'ask-ai-field-hint';
    const result = await fetchModelList();
    btn.classList.remove('loading');
    if (result.error) {
      sel.hidden = true; inp.hidden = false;
      hint.textContent = '拉取失败: ' + result.error + ' — 已回退手填模式';
      hint.className = 'ask-ai-field-hint err';
      return;
    }
    // populate dropdown
    sel.innerHTML = result.models.map(m =>
      '<option value="' + escapeHtml(m) + '">' + escapeHtml(m) + '</option>'
    ).join('');
    // 保留当前选中的 model 如果在列表里
    const currentModel = (S.config && S.config.model) || inp.value.trim();
    if (currentModel && result.models.includes(currentModel)) sel.value = currentModel;
    sel.hidden = false; inp.hidden = true;
    hint.textContent = '✓ 已拉到 ' + result.models.length + ' 个 model';
    hint.className = 'ask-ai-field-hint ok';
    // 缓存给 in-panel 快切器
    if (S.config) S.config.cachedModels = result.models;
  }
  async function testConfig() {
    const cfg = collectConfig();
    const status = document.getElementById('cfg-status');
    status.textContent = '测试中...';
    status.className = 'ask-ai-config-status';
    try {
      const url = cfg.baseUrl.replace(/\/+$/, '') + '/chat/completions';
      const resp = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + cfg.apiKey,
        },
        body: JSON.stringify({
          model: cfg.model,
          messages: [{role: 'user', content: 'hi'}],
          max_tokens: 2,
          stream: false,
        }),
      });
      if (resp.ok) {
        status.textContent = '✓ 连接成功';
        status.className = 'ask-ai-config-status ok';
      } else {
        const text = await resp.text();
        status.textContent = '✗ HTTP ' + resp.status + ': ' + text.slice(0, 120);
        status.className = 'ask-ai-config-status err';
      }
    } catch (e) {
      status.textContent = '✗ ' + e.message + (String(e).includes('CORS') || String(e).includes('Failed to fetch')
        ? ' (可能是 CORS 问题 — 试着用 python -m http.server 起本地 server 再访问)' : '');
      status.className = 'ask-ai-config-status err';
    }
  }

  // ---- Prompt construction ----
  const DEFAULT_SYSTEM_PROMPT =
      '你是一个 ML 论文/源码理解助手, 帮用户阅读一份 SmolVLA 训练 trace 的可交互架构网页。\n\n'
    + '**重要: 你收到的不是 HTML 源代码** — 而是从那个页面抽出的**结构化 context**:\n'
    + '1. 用户选中的文本片段 (★ 最高优先级, 如果有)\n'
    + '2. 用户当前展开看的模块的完整详情 (qualname / shape / what / formula / 源码位置)\n'
    + '3. 页面整体拓扑 (顶层模块图 + 数据流方向 + 关键 shape) — 背景参考\n'
    + '4. 用户当前的 breadcrumb 路径 (位置 hint)\n\n'
    + '**主语判定规则 (重要)**:\n'
    + '- 如果 context 里有"选中的文本"段, **用户问题里的"这/this/它/这段/这个"默认指代那段选中文本**, 不是页面整体, 也不是当前模块。例: 选中"prefix 是模型能看到的那段 token" + 问"这是什么意思" → 回答"prefix 是什么意思", 不是"这一步是什么意思"。\n'
    + '- 如果没有选中, 主语才落到"当前展开的模块"或"breadcrumb 位置"上。\n\n'
    + '**不要在回答 / 思考里说"我没有 HTML 内容"这种话**。你有的就是上面四类结构化片段, 比 HTML 干净得多。\n'
    + 'context 没覆盖的细节: 提醒用户**点开对应模块** (drawer 展开后下一轮自动注入), 或基于通用知识直接答。\n\n'
    + '**回答风格要求**:\n'
    + '- 用 markdown: 代码块 ```...```, 行内 `code`, **粗体**, `-` / `1.` 列表\n'
    + '- 段落紧凑, **不要每句话单独成段**; 长答案才用 `###` 小标题分段, 短答案直接一两段就够\n'
    + '- 数学公式: 行内 `\\(x_t\\)` 块级 `$$...$$` (KaTeX 语法)\n'
    + '- 简洁直接、不套话; 跟当前位置无关的问题正常回答, 不强行联系当前位置';

  function buildSystemPrompt() {
    if (S.config && S.config.systemPrompt && S.config.systemPrompt.trim()) {
      return S.config.systemPrompt.trim();
    }
    return DEFAULT_SYSTEM_PROMPT;
  }

  // ---- 页面拓扑概览 (每次第一轮塞进 context, 让 AI 不会嘀咕"哪里有 HTML") ----
  function buildTopologyOverview() {
    if (typeof TREE === 'undefined' || !TREE.pipeline) return '';
    const p = TREE.pipeline;
    const lines = [];
    if (p.name) lines.push('页面: ' + p.name + (p.sub ? ' (' + p.sub + ')' : ''));
    if (p.story_line) lines.push('概念: ' + stripHtml(p.story_line));
    lines.push('');
    lines.push('拓扑 (每个 stage 一行, "/" 隔开同 stage 的并行节点):');
    const stages = p.stages || [];
    stages.forEach((stage, i) => {
      const nodeNames = (stage.nodes || []).map(id => {
        const n = TREE[id];
        if (!n) return id;
        const desc = n.sub ? ' [' + n.sub + ']' : '';
        return (n.name || id) + desc;
      }).join(' / ');
      let flag = '';
      if (stage.is_heart) flag = ' ★心脏模块';
      else if (stage.training_only) flag = ' (训练 only)';
      lines.push('  ' + (i + 1) + '. ' + nodeNames + flag);
      if (stage.outgoing_labels) {
        Object.entries(stage.outgoing_labels).forEach(([from, label]) => {
          lines.push('      ↓ ' + label);
        });
      }
    });
    return lines.join('\n');
  }
  let _topologyCache = null;
  function getTopologyOverview() {
    if (_topologyCache === null) _topologyCache = buildTopologyOverview();
    return _topologyCache;
  }

  function snapshotContext(opts) {
    opts = opts || {};
    const breadcrumb = state.path.map(p => (TREE[p] && TREE[p].name) || p).join(' ▸ ');
    const moduleId = opts.moduleId || state.drawerOpen;
    let ctx = '';

    // === 优先级 1: 选中文本 (如果有) ===
    // 这是用户问"这是什么意思"时的真正主语, 必须放最前面 + 明确标重点。
    // 之前的版本把选中文本放最后, 导致 AI 容易把"这"理解成"整个页面"。
    if (S.selection) {
      ctx += '## ⚠️ 用户选中的文本 (用户问题的主要主语 — "这/this/它" 通常指它)\n';
      ctx += '---SELECTION START---\n' + S.selection + '\n---SELECTION END---\n\n';
    }

    // === 优先级 2: 用户当前展开的模块详情 ===
    if (moduleId && TREE[moduleId]) {
      const n = TREE[moduleId];
      ctx += '## 用户当前展开看的模块\n';
      ctx += '名称: ' + (n.name || moduleId) + (n.sub ? ' · ' + n.sub : '') + '\n';
      ctx += '类型: ' + n.type + (n.functional ? ' (functional)' : '') + '\n';
      ctx += 'Qualname: ' + (n._qualname || n.qualname || '-') + '\n';
      if (n._callCount && n._callCount > 1) ctx += '调用次数: × ' + n._callCount + '\n';
      ctx += 'Shape: ' + stripHtml(n._shape_html || n.shape_html || '-') + '\n';
      if (n.what) ctx += 'What: ' + stripHtml(n.what) + '\n';
      if (n.formula) ctx += 'Formula (LaTeX): ' + n.formula + '\n';
      if (n.callout) ctx += '备注: ' + stripHtml(n.callout) + '\n';
      ctx += '源码: ' + (n._src || n.src || '-') + '\n\n';
    }

    // === 优先级 3: 整体拓扑 (背景知识, 让 AI 知道页面长什么样) ===
    const topo = getTopologyOverview();
    if (topo) ctx += '## 页面整体拓扑 (背景参考)\n' + topo + '\n\n';

    // === 优先级 4: 用户当前 breadcrumb (位置信息, 最弱的 hint) ===
    ctx += '## 用户当前位置\n';
    ctx += 'Breadcrumb: ' + breadcrumb + '\n';
    return ctx;
  }

  // ---- UI helpers ----
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function clearEmptyState() {
    const msgs = document.getElementById('ask-ai-messages');
    const empty = msgs.querySelector('.ask-ai-empty');
    if (empty) empty.remove();
  }
  function appendMessage(role, text) {
    clearEmptyState();
    const msgs = document.getElementById('ask-ai-messages');
    const wrap = document.createElement('div');
    wrap.className = 'ask-msg ' + role;
    wrap.innerHTML = '<div class="ask-msg-role">' + (role === 'user' ? 'YOU' : 'AI') + '</div>'
                   + '<div class="ask-msg-body"></div>';
    wrap.querySelector('.ask-msg-body').textContent = text || '';
    msgs.appendChild(wrap);
    msgs.scrollTop = msgs.scrollHeight;
    return wrap;
  }
  function appendError(msg, withSettings) {
    clearEmptyState();
    const msgs = document.getElementById('ask-ai-messages');
    const div = document.createElement('div');
    div.className = 'ask-error';
    div.innerHTML = escapeHtml(msg)
      + (withSettings ? ' <button onclick="window.AskAI && AskAI.openConfig()">检查设置</button>' : '');
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }
  function renderAssistantMarkdown(el, text) {
    // 较完整的 markdown: 代码块 / 标题 / 列表 / 行内 / 段落 / blockquote / hr + KaTeX 数学
    let html = escapeHtml(text || '');

    // 1. 抽出 fenced code blocks (避免内部被 markdown 二次处理)
    const codeBlocks = [];
    html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, function(_m, lang, code) {
      codeBlocks.push('<pre' + (lang ? ' class="lang-' + lang + '"' : '') + '><code>' + code + '</code></pre>');
      return ' CODEBLOCK_' + (codeBlocks.length - 1) + ' ';
    });

    // 2. 行级转换: header / hr / blockquote / list
    const lines = html.split('\n');
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      // 标题
      const hm = line.match(/^(#{1,6})\s+(.+)$/);
      if (hm) {
        const lvl = Math.min(hm[1].length, 4);  // 限制最多 h4
        out.push('<h' + lvl + '>' + hm[2] + '</h' + lvl + '>');
        i++; continue;
      }
      // 水平线
      if (/^(---|\*\*\*|___)\s*$/.test(line)) {
        out.push('<hr>');
        i++; continue;
      }
      // 无序列表 - 或 * (允许嵌套缩进, 但简化处理为扁平 ul)
      if (/^[\-\*]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^[\-\*]\s+/.test(lines[i])) {
          items.push('<li>' + lines[i].replace(/^[\-\*]\s+/, '') + '</li>');
          i++;
        }
        out.push('<ul>' + items.join('') + '</ul>');
        continue;
      }
      // 有序列表 1. 2. 3.
      if (/^\d+\.\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
          items.push('<li>' + lines[i].replace(/^\d+\.\s+/, '') + '</li>');
          i++;
        }
        out.push('<ol>' + items.join('') + '</ol>');
        continue;
      }
      // Blockquote
      if (/^&gt;\s+/.test(line) || /^>\s+/.test(line)) {
        const quotes = [];
        while (i < lines.length && (/^&gt;\s+/.test(lines[i]) || /^>\s+/.test(lines[i]))) {
          quotes.push(lines[i].replace(/^(&gt;|>)\s+/, ''));
          i++;
        }
        out.push('<blockquote>' + quotes.join('<br>') + '</blockquote>');
        continue;
      }
      // 空行 = 段落分隔 (多个连续空行折叠成一个分隔, 不输出空字符串避免空 <p>)
      if (!line.trim()) {
        i++;
        // skip 连续多个空行
        while (i < lines.length && !lines[i].trim()) i++;
        continue;
      }
      // 普通文本行 — 累积进段落
      const para = [line];
      i++;
      while (i < lines.length && lines[i].trim()
             && !/^(#{1,6}\s|[\-\*]\s|\d+\.\s|&gt;\s|>\s)/.test(lines[i])
             && !/^(---|\*\*\*|___)\s*$/.test(lines[i])
             && !lines[i].includes(' CODEBLOCK_')) {
        para.push(lines[i]);
        i++;
      }
      out.push('<p>' + para.join('<br>') + '</p>');
    }
    html = out.join('\n');

    // 3. 行内: 粗体 / 斜体 / inline code (注意顺序: code 最先, 避免内部被加粗)
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/(?<!\w)\*([^*\n]+)\*(?!\w)/g, '<em>$1</em>');
    // 链接 [text](url)
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // 4. 还原 code blocks
    html = html.replace(/ CODEBLOCK_(\d+) /g, function(_m, idx) { return codeBlocks[+idx] || ''; });

    el.innerHTML = html;
    // 5. KaTeX 渲染数学
    if (window.renderMathInElement) {
      try {
        renderMathInElement(el, {
          delimiters: [{left: '$$', right: '$$', display: true}, {left: '\\(', right: '\\)', display: false}],
          throwOnError: false,
        });
      } catch (e) {}
    }
  }

  // ---- 错误诊断: 把 HTTP 错误码 / fetch 异常 翻译成可操作的中文提示 ----
  function diagnoseHttpError(status, body, endpoint) {
    if (status === 401) return 'API key 无效或过期。请点上方 ⚙ 在设置里更新 key。';
    if (status === 403) return 'API 拒绝访问 (403)。可能 key 没启用对应 model, 或 region 受限。';
    if (status === 404) return 'Endpoint 不存在: ' + endpoint + '\n检查 Base URL 是否正确。';
    if (status === 429) return '触发限流 (429)。等几秒重试, 或换 model / 升级 plan。';
    if (status >= 500) return '服务器错误 ' + status + '。endpoint 暂时不可用, 稍后重试。';
    const trimmed = String(body || '').trim();
    return 'HTTP ' + status + (trimmed ? ': ' + trimmed.slice(0, 240) : '');
  }
  function diagnoseFetchError(err, endpoint) {
    const msg = String(err.message || err);
    const protocol = location.protocol;
    // 最高频的首次失败: file:// 撞 CORS
    if (protocol === 'file:' && (msg.includes('Failed to fetch') || msg.includes('NetworkError'))) {
      return '看起来你是双击 HTML 用 file:// 打开的, 浏览器禁止从 file:// 直接请求外部 API。\n'
           + '解决: 起一个本地 server, 例如\n'
           + '  cd <html 所在目录> && python -m http.server 8000\n'
           + '然后用 http://localhost:8000/<filename>.html 访问。';
    }
    if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
      if (endpoint.includes('localhost') || endpoint.includes('127.0.0.1')) {
        return '本地 endpoint 不可达: ' + endpoint + '\n确认 Ollama / 本地 server 真的在跑 (例: ollama serve)。';
      }
      return '请求发不出: ' + msg + '\n常见原因: endpoint 拒绝浏览器直连 CORS / 网络不可达 / endpoint 输错。';
    }
    return msg;
  }

  // ---- 流式 KaTeX 节流: 期间 150ms 才重渲一次 markdown + 数学, 中间用 textContent ----
  const KATEX_THROTTLE_MS = 150;

  // ---- In-panel model quickswitch ----
  function toggleModelDropdown(e) {
    if (e) e.stopPropagation();
    const dd = document.getElementById('model-dropdown');
    if (!dd.hidden) { dd.hidden = true; return; }
    if (!S.config) { openConfig(true); return; }
    const models = (S.config.cachedModels && S.config.cachedModels.length)
                   ? S.config.cachedModels
                   : (S.config.model ? [S.config.model] : []);
    if (!models.length) {
      dd.innerHTML = '<div class="ask-model-dropdown-empty">无 model 列表 — 去 ⚙ 设置里点 ↻ 拉取</div>';
    } else {
      dd.innerHTML = models.map(m =>
        '<button type="button" data-model="' + escapeHtml(m) + '"' + (m === S.config.model ? ' class="current"' : '') + '>'
        + escapeHtml(m) + '</button>'
      ).join('');
      dd.querySelectorAll('button[data-model]').forEach(btn => {
        btn.addEventListener('click', () => attemptModelSwitch(btn.dataset.model));
      });
    }
    dd.hidden = false;
  }

  function attemptModelSwitch(newModel) {
    const dd = document.getElementById('model-dropdown');
    dd.hidden = true;
    if (!S.config || newModel === S.config.model) return;
    if (S.messages.length === 0) {
      // 没历史, 直接切
      _applyModelSwitch(newModel, false);
      return;
    }
    // 有历史 — 弹小确认条问是否清空
    const switcher = document.getElementById('model-switcher');
    const confirm = document.createElement('div');
    confirm.className = 'ask-switch-confirm';
    confirm.innerHTML =
      '切换到 <strong>' + escapeHtml(newModel) + '</strong> 后, 之前的对话上下文在新模型里可能表现异常。'
      + '<div class="ask-switch-confirm-actions">'
      + '  <button type="button" class="primary" data-action="clear">清空 + 切换</button>'
      + '  <button type="button" data-action="keep">保留对话</button>'
      + '  <button type="button" data-action="cancel">取消</button>'
      + '</div>';
    switcher.appendChild(confirm);
    confirm.querySelectorAll('button').forEach(b => {
      b.addEventListener('click', () => {
        const a = b.dataset.action;
        confirm.remove();
        if (a === 'cancel') return;
        if (a === 'clear') resetConversation();
        _applyModelSwitch(newModel, a === 'keep');
      });
    });
  }

  function _applyModelSwitch(newModel, keepHistory) {
    S.config.model = newModel;
    saveConfig(S.config);
    if (keepHistory) {
      flashToast('已切到 ' + newModel + ' (对话历史保留, 注意新模型可能不一致)', false);
    } else {
      flashToast('已切到 ' + newModel);
    }
  }

  // ---- Send button mode (send / stop) ----
  function setSendButtonMode(mode) {
    const btn = document.getElementById('ask-ai-send');
    if (!btn) return;
    btn.dataset.mode = mode;
    if (mode === 'stop') {
      btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><rect x="4" y="4" width="8" height="8" rx="1.2"/></svg>';
      btn.title = '停止生成';
      btn.classList.add('stop-mode');
    } else {
      btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 8 L14 2 L11 14 L8 9 L2 8 Z"/></svg>';
      btn.title = '发送 (Ctrl+Enter)';
      btn.classList.remove('stop-mode');
    }
  }

  // ---- Send message (streaming) ----
  async function sendMessage(userText) {
    if (S.isStreaming) return;
    if (!userText || !userText.trim()) return;
    if (!S.config) { openConfig(true); return; }

    // ---- Compact context: 第一轮发完整, 之后只在 context 变化时重发 ----
    const isFirst = S.messages.length === 0;
    const fullCtx = snapshotContext({moduleId: state.drawerOpen});
    // djb2 hash
    let h = 5381;
    for (let i = 0; i < fullCtx.length; i++) h = ((h << 5) + h) + fullCtx.charCodeAt(i);
    const ctxHash = String(h);
    let fullUserContent;
    if (isFirst || ctxHash !== S.lastSentContextHash) {
      fullUserContent = fullCtx + '\n## 用户的问题\n' + userText;
      S.lastSentContextHash = ctxHash;
    } else {
      // 后续轮且 context 未变: 只发问题, 节省 token
      fullUserContent = userText;
    }

    S.messages.push({role: 'user', content: fullUserContent});

    appendMessage('user', userText);  // UI 只显示用户原文, 不显示注入的 context
    const assistantEl = appendMessage('assistant', '');
    assistantEl.classList.add('streaming');
    // 准备 thinking 区 + content 区 (即使 provider 不返回 thinking, 也无害)
    const bodyEl = assistantEl.querySelector('.ask-msg-body');
    bodyEl.innerHTML =
        '<details class="ask-thinking" id="" hidden><summary>思考过程</summary><div class="ask-thinking-content"></div></details>'
      + '<div class="ask-content"></div>';
    const thinkingEl = bodyEl.querySelector('.ask-thinking');
    const thinkingContentEl = bodyEl.querySelector('.ask-thinking-content');
    const contentEl = bodyEl.querySelector('.ask-content');

    // ---- 对话保留轮数 (retention) — 仅截 history, 当前轮一定要送 ----
    let convoMessages = S.messages;
    const retention = S.config.retention != null ? S.config.retention : 5;
    if (retention > 0) {
      // 一"轮" = user + assistant 各一条, 所以保留 retention*2 条 + 当前 user 这条
      // 取 messages 末尾的 retention*2 + 1 条 (当前 user 永远在末尾)
      const keep = retention * 2 + 1;
      if (S.messages.length > keep) convoMessages = S.messages.slice(-keep);
    }

    const url = S.config.baseUrl.replace(/\/+$/, '') + '/chat/completions';
    const payload = {
      model: S.config.model,
      messages: [
        {role: 'system', content: buildSystemPrompt()},
        ...convoMessages,
      ],
      stream: true,
      temperature: S.config.temperature != null ? S.config.temperature : 0.4,
      top_p: S.config.topP != null ? S.config.topP : 1.0,
      max_tokens: S.config.maxTokens != null ? S.config.maxTokens : 8192,
    };

    S.isStreaming = true;
    S.abortCtrl = new AbortController();
    setSendButtonMode('stop');  // 切到"停止生成"按钮
    let fullText = '';
    let thinkingText = '';
    let lastKatexRender = 0;

    try {
      const resp = await fetch(url, {
        method: 'POST',
        signal: S.abortCtrl.signal,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + (S.config.apiKey || ''),
        },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const text = await resp.text();
        assistantEl.remove();
        appendError(diagnoseHttpError(resp.status, text, url), true);
        return;
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;  // SSE event separator (空行)
          // 跳过 SSE meta fields: retry: / id: / event: / : (comment)
          // —— 经过第三方网关或本地 proxy 时这些字段都可能出现, 不能无脑 JSON.parse
          if (trimmed.startsWith(':')) continue;
          if (trimmed.startsWith('retry:') || trimmed.startsWith('id:') || trimmed.startsWith('event:')) continue;
          if (!trimmed.startsWith('data:')) continue;
          const data = trimmed.slice(5).trim();
          if (!data || data === '[DONE]') continue;
          try {
            const parsed = JSON.parse(data);
            const choice = parsed.choices && parsed.choices[0];
            if (!choice) continue;
            const delta = choice.delta || {};
            // 思考过程: DeepSeek = reasoning_content, Anthropic = thinking, Gemini = thoughts (有时)
            const thinkChunk = delta.reasoning_content || delta.thinking || delta.thoughts;
            const contentChunk = delta.content;
            if (thinkChunk) {
              thinkingText += thinkChunk;
              thinkingEl.hidden = false;
              thinkingEl.open = true;  // 流式时展开
              thinkingContentEl.textContent = thinkingText;
            }
            if (contentChunk) {
              fullText += contentChunk;
              // 思考结束 + 正文开始 → 收起 thinking
              if (thinkingEl && !thinkingEl.dataset.collapsedOnce && thinkingText) {
                thinkingEl.open = false;
                thinkingEl.dataset.collapsedOnce = '1';
              }
              // 即时显示纯文本 (cheap)
              contentEl.textContent = fullText;
              const msgs = document.getElementById('ask-ai-messages');
              msgs.scrollTop = msgs.scrollHeight;
              // markdown + KaTeX 节流到 150ms 一次
              const now = performance.now();
              if (now - lastKatexRender > KATEX_THROTTLE_MS) {
                lastKatexRender = now;
                renderAssistantMarkdown(contentEl, fullText);
              }
            }
          } catch (e) { /* partial json across chunks, skip */ }
        }
      }
      // streaming done — 最终一次完整渲染 (确保所有公式 + 代码块 + delimiter 都闭合渲染)
      assistantEl.classList.remove('streaming');
      if (fullText) renderAssistantMarkdown(contentEl, fullText);
      // thinking 区流完后默认折叠 (用户想看可手动展开)
      if (thinkingText) thinkingEl.open = false;
      else thinkingEl.hidden = true;
      S.messages.push({role: 'assistant', content: fullText});
    } catch (e) {
      assistantEl.classList.remove('streaming');
      if (e.name === 'AbortError') {
        // 用户主动点了"停止" — 保留已收到的内容, 加个标记
        renderAssistantMarkdown(contentEl, (fullText || '') + '\n\n_[已停止生成]_');
        if (thinkingText) { thinkingEl.open = false; } else { thinkingEl.hidden = true; }
        if (fullText) S.messages.push({role: 'assistant', content: fullText});
      } else {
        assistantEl.remove();
        appendError(diagnoseFetchError(e, url), true);
      }
    } finally {
      S.isStreaming = false;
      S.abortCtrl = null;
      setSendButtonMode('send');  // 恢复发送按钮
    }
  }

  // ---- Wire up event listeners ----
  function init() {
    S.config = loadConfig();
    updateModelTag();
    setSendButtonMode('send');  // 确保 dataset.mode = 'send' 初始值

    // Selection -> show fab (统一走 recheckSelection, debounce 100ms)
    // selectionchange 是主控 (任何方式的选区变化都触发), mouseup/keyup 兜底
    document.addEventListener('selectionchange', recheckSelection);
    document.addEventListener('mouseup', recheckSelection);
    document.addEventListener('keyup', e => {
      // Shift+arrow 等键盘选区
      if (e.shiftKey || e.key === 'ArrowLeft' || e.key === 'ArrowRight'
          || e.key === 'ArrowUp' || e.key === 'ArrowDown') recheckSelection();
    });
    document.addEventListener('scroll', hideFab, true);

    const fab = document.getElementById('ask-ai-fab');
    fab.addEventListener('mousedown', e => e.preventDefault());  // don't lose selection
    fab.addEventListener('click', () => { hideFab(); openPanel({}); });

    // Panel controls
    document.getElementById('ask-ai-close').addEventListener('click', closePanel);
    document.getElementById('ask-ai-reset').addEventListener('click', resetConversation);
    document.getElementById('ask-ai-settings').addEventListener('click', () => openConfig(false));

    // Submit (按钮在 stop 模式时点击 = 中止流式; 否则 = 发送)
    document.getElementById('ask-ai-form').addEventListener('submit', e => {
      e.preventDefault();
      const btn = document.getElementById('ask-ai-send');
      if (btn && btn.dataset.mode === 'stop' && S.abortCtrl) {
        S.abortCtrl.abort();
        return;
      }
      const input = document.getElementById('ask-ai-input');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      sendMessage(text);
    });
    document.getElementById('ask-ai-input').addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        document.getElementById('ask-ai-form').dispatchEvent(new Event('submit', {cancelable: true}));
      }
    });

    // Config modal
    document.getElementById('ask-ai-config-close').addEventListener('click', closeConfig);
    document.getElementById('ask-ai-config-backdrop').addEventListener('click', closeConfig);
    document.getElementById('cfg-provider').addEventListener('change', e => applyProviderPreset(e.target.value));
    document.getElementById('cfg-model-refresh').addEventListener('click', refreshModels);
    document.getElementById('cfg-test').addEventListener('click', testConfig);
    document.getElementById('cfg-save').addEventListener('click', () => {
      const cfg = collectConfig();
      if (!cfg.baseUrl || !cfg.model) {
        const status = document.getElementById('cfg-status');
        status.textContent = '✗ Base URL 和 Model 必填';
        status.className = 'ask-ai-config-status err';
        return;
      }
      saveConfig(cfg);
      closeConfig();
      flashToast('API 配置已保存');
    });

    // In-panel model quickswitch
    document.getElementById('ask-ai-model-tag').addEventListener('click', toggleModelDropdown);
    // 点击 dropdown 外部 → 关闭
    document.addEventListener('click', e => {
      const switcher = document.getElementById('model-switcher');
      if (switcher && !switcher.contains(e.target)) {
        const dd = document.getElementById('model-dropdown');
        if (dd && !dd.hidden) dd.hidden = true;
      }
    });
  }

  return {
    init: init,
    openPanel: openPanel,
    closePanel: closePanel,
    openConfig: function() { openConfig(false); },
    get state() { return S; },
  };
})();

// Init after DOM ready
if (document.readyState !== 'loading') AskAI.init();
else document.addEventListener('DOMContentLoaded', AskAI.init);
"""


def render(trace_path: Path, out_path: Path) -> None:
    trace = load_trace(trace_path)
    calls_by_name = pair_events_by_name(trace)
    enriched_tree = enrich_with_trace(TREE, calls_by_name)

    tree_json = json.dumps(enriched_tree, ensure_ascii=False)
    js = JS_TEMPLATE.replace("__TREE_JSON__", tree_json)

    n_fwd = sum(1 for e in trace["module_events"] if e["phase"] == "forward_post")
    n_bwd = sum(1 for e in trace["module_events"] if e["phase"] == "backward")
    n_sdpa = len(trace.get("functional_events", []))
    total_params = 0
    root = next((v for v in trace.get("module_tree", {}).values() if v["name"] == "<root>"), None)
    if root:
        total_params = root["num_params_total"]

    header_html = (
        f"<h1>SmolVLA · 一次训练 step 的可交互架构</h1>"
        f'<div class="lede">'
        f'trace: <b>{html.escape(trace_path.name)}</b><span class="dot">·</span>'
        f'params: <b>{total_params/1e6:.0f}M</b><span class="dot">·</span>'
        f'forwards: <b>{n_fwd}</b><span class="dot">·</span>'
        f'backward grads: <b>{n_bwd}</b><span class="dot">·</span>'
        f'SDPA ops: <b>{n_sdpa}</b>'
        f"</div>"
    )

    legend_html = (
        '<div class="legend">'
        '<span class="legend-item"><span class="swatch composite"></span>composite · click to drill in</span>'
        '<span class="legend-item"><span class="swatch leaf"></span>leaf · click for details</span>'
        '<span class="legend-item"><span class="swatch functional"></span>functional (not nn.Module)</span>'
        '<span class="legend-item">hover for tooltip</span>'
        '<span class="legend-item"><kbd>Esc</kbd> close drawer / pop breadcrumb</span>'
        '</div>'
    )

    body = (
        f'<div class="container">'
        f'{header_html}'
        f'<div class="canvas-wrap">'
        f'<div class="breadcrumb" id="breadcrumb"></div>'
        f'<div class="blurb" id="blurb"></div>'
        f'<div class="canvas" id="canvas"></div>'
        f'{legend_html}'
        f'</div>'
        f'</div>'
        f'<div class="drawer-backdrop" id="backdrop"></div>'
        f'<aside class="drawer" id="drawer" aria-hidden="true">'
        f'<button class="drawer-close" id="drawer-close" aria-label="Close">×</button>'
        f'<div id="drawer-content"></div>'
        f'</aside>'
    )

    out_html = (
        f"<!doctype html><html lang='zh-CN'><head>"
        f"<meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>SmolVLA Trace · {html.escape(trace_path.name)}</title>"
        f"{FONT_AND_KATEX}"
        f"<style>{CSS}\n{ASK_AI_CSS}</style>"
        f"</head><body>"
        f"{body}"
        f"{ASK_AI_HTML}"
        f"<script>{js}</script>"
        f"<script>{ASK_AI_JS}</script>"
        f"</body></html>"
    )
    out_path.write_text(out_html, encoding="utf-8")
    print(f"rendered → {out_path}  ({len(out_html)//1024}KB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_json", type=Path)
    ap.add_argument("output_html", type=Path, nargs="?", default=None)
    args = ap.parse_args()
    out = args.output_html or args.trace_json.with_suffix(".html")
    render(args.trace_json, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
