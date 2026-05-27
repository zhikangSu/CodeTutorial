"""Enrich tree with real trace data (shapes, src, call counts, backward grads)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .source_links import SMOLVLA_DECL_SRC, vscode_link, activation_svg
from .trace_utils import find_tensor_shape, find_tensor_dtype, fmt_arrow


def _backward_counts(trace: dict) -> Counter[str]:
    """Count backward events per module dotted name, so leaves can show
    a trained-vs-frozen badge in the drawer."""
    mt = trace.get("module_tree", {})
    id_to_name = {int(k): v["name"] for k, v in mt.items()}
    counts: Counter[str] = Counter()
    for e in trace.get("module_events", []):
        if e.get("phase") != "backward":
            continue
        nm = id_to_name.get(e.get("module_id"))
        if nm:
            counts[nm] += 1
    return counts


def enrich_with_trace(tree: dict, trace: dict) -> dict:
    """Fill TREE nodes with real shape / src / callCount / backward info.

    `trace` is the full trace JSON; we derive calls_by_name and backward
    counts internally so the caller doesn't have to.
    """
    from .trace_utils import pair_events_by_name
    calls_by_name = pair_events_by_name(trace)
    bwd_counts = _backward_counts(trace)

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
                #   否则用 trace 抓到的 src_file (forward 实际执行的位置)
                static_src = node.get("src")
                if static_src and static_src != "—":
                    node["_src"] = static_src
                    node["_src_link"] = vscode_link(static_src)
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
        if node.get("act_chart"):
            node["_act_chart_svg"] = activation_svg(node["act_chart"])

        # Backward / training state — only meaningful for leaves with trace_name
        if node.get("type") == "leaf" and node.get("trace_name"):
            tn = node["trace_name"]
            # For repeated layers, sum backward across all layers
            if ".layers.0." in tn:
                pattern = re.sub(r"\.layers\.0\.", r".layers.\\d+.", re.escape(tn))
                regex = re.compile("^" + pattern + "$")
                bwd_n = sum(c for n, c in bwd_counts.items() if regex.match(n))
            else:
                bwd_n = bwd_counts.get(tn, 0)
            if bwd_n > 0:
                node["_trained"] = True
                node["_backward_count"] = bwd_n
            elif not node.get("functional"):
                # Only mark "frozen" for real nn.Modules; functional ops never
                # have parameter gradients regardless of training mode.
                node["_trained"] = False

        enriched[nid] = node
    return enriched
