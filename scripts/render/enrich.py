"""Enrich tree with real trace data (shapes, src, call counts)."""

from __future__ import annotations

import re
from typing import Any

from .source_links import SMOLVLA_DECL_SRC, vscode_link, activation_svg
from .trace_utils import find_tensor_shape, find_tensor_dtype, fmt_arrow


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
        enriched[nid] = node
    return enriched
