"""render_html.py — SmolVLA trace JSON → 可交互架构图 HTML.

实现的 UX:
- 首屏一张方块架构图, 左→右数据流
- composite 节点点击 = 主图替换为子节点
- leaf 节点点击 = 右侧抽屉详情
- 重复层（VLM 16 / Expert 16 / Vision 12）合并为单节点 + ×N badge
- shape 用语义维度名 ([batch=16, seq=177, hidden=960])
- Esc 收抽屉 / 回 breadcrumb

数据 + 资源拆分:
- render/tree_smolvla.py:    SmolVLA-specific TREE (~700 行)
- render/source_links.py:    PATH_MAP / SMOLVLA_DECL_SRC / vscode_link
- render/trace_utils.py:     load_trace / fmt_shape / pair_events_by_name
- render/enrich.py:          把 trace JSON 真实 shape/src 填进 TREE
- render/assets/*.css|js|html: CSS / JS / HTML 抽出成真文件

用法:
    python render_html.py <trace.json> [<output.html>]
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from render import (
    TREE,
    bundle_assets,
    enrich_with_trace,
    load_trace,
    pair_events_by_name,
)


def render(trace_path: Path, out_path: Path) -> None:
    trace = load_trace(trace_path)
    calls_by_name = pair_events_by_name(trace)
    enriched_tree = enrich_with_trace(TREE, calls_by_name)

    assets = bundle_assets()
    tree_json = json.dumps(enriched_tree, ensure_ascii=False)
    main_js = assets["main_js"].replace("__TREE_JSON__", tree_json)

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
        f"{assets['katex_head']}"
        f"<style>{assets['main_css']}\n{assets['ask_ai_css']}</style>"
        f"</head><body>"
        f"{body}"
        f"{assets['ask_ai_html']}"
        f"<script>{main_js}</script>"
        f"<script>{assets['ask_ai_js']}</script>"
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
