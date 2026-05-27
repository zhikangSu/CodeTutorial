"""Source-link helpers: PATH_MAP, SMOLVLA_DECL_SRC, vscode_link, activation_svg.

NOTE (P1 work): PATH_MAP + SMOLVLA_DECL_SRC are SmolVLA-specific hardcodes;
making them pluggable for other PyTorch repos is the next milestone.
"""

from __future__ import annotations

import math


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
    if rel_path.startswith("/"):
        return f"vscode://file{rel_path}:{lno}"
    for prefix, abs_prefix in PATH_MAP.items():
        if rel_path.startswith(prefix):
            return f"vscode://file{abs_prefix}{rel_path[len(prefix):]}:{lno}"
    return None


def activation_svg(kind: str) -> str:
    """生成 GELU / SiLU 等激活函数的小 SVG 曲线, 嵌进 drawer 帮助直观看图。"""
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
