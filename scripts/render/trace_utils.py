"""trace JSON helpers + semantic dimension labeling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
