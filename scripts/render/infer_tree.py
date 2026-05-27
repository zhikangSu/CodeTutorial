"""infer_tree.py — 从 trace JSON 推 TREE 骨架, 让换 repo 半自动化。

机械化部分（这个脚本做）:
1. PATH_MAP   — 扫所有 src_file 路径, 找 package 根（lerobot/, transformers/, torch/ 等）
2. DECL_SRC   — 用 AST 找每个 leaf 模块的"声明行"（parent class 里 self.X = nn.Linear(...) 那一行）,
                 替换掉 trace 抓的 forward 实际执行位置（torch 内部）
3. TREE 骨架  — 从 module_tree 拼出 composite/leaf 层级, 合并 .layers.N. 重复块,
                 用真实 shape 填 in_hint/out_hint 占位
4. digest.md  — 给 Claude 一份易读的 markdown 摘要, 用于补语义字段

语义化部分（Claude 在对话里做）:
- 每个 node 的 concept_name / sub / tooltip / what / blurb / formula / callout
- pipeline.stages 的 Y-shape 拓扑重构（infer 默认输出 linear stages）
- is_heart / training_only 标记
- story_line

用法:
    python -m render.infer_tree <trace.json> [--out-dir inferred/]
    # 然后 Claude 读 inferred/digest.md + 论文 + tree_skeleton.py, 改成最终 TREE
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .trace_utils import find_tensor_shape, find_tensor_dtype, pair_events_by_name


# ============================================================================
# Known leaf classes (no children needed; treat as atomic leaves)
# ============================================================================
LEAF_CLASSES = {
    "Linear", "Conv1d", "Conv2d", "Conv3d", "Embedding",
    "LayerNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "GroupNorm", "InstanceNorm1d", "InstanceNorm2d",
    "LlamaRMSNorm", "RMSNorm", "T5LayerNorm",
    "GELU", "GELUTanh", "ReLU", "SiLU", "SiLUActivation", "Tanh", "Sigmoid",
    "Dropout", "Identity",
    "LlamaRotaryEmbedding", "RotaryEmbedding",
}


# ============================================================================
# Step 1: PATH_MAP — scan src_file paths, find package roots
# ============================================================================
def infer_path_map(trace: dict) -> dict[str, str]:
    """Find common package roots from all src_file paths in trace.

    Looks for path segments after standard markers (`site-packages/`, local repo
    layouts) and groups them into prefix → absolute-root mappings.
    """
    paths = set()
    for v in trace.get("module_tree", {}).values():
        if v.get("src_file"):
            paths.add(v["src_file"])
    for e in trace.get("module_events", []):
        if e.get("src_file"):
            paths.add(e["src_file"])

    candidates: dict[str, set[str]] = defaultdict(set)
    for p in paths:
        # site-packages: /…/site-packages/<pkg>/…
        m = re.search(r"/site-packages/([^/]+)/", p)
        if m:
            pkg = m.group(1)
            abs_prefix = p[:m.end()]
            candidates[f"{pkg}/"].add(abs_prefix)
            continue
        # local repo style: /…/<reponame>/src/<pkg>/…  (the lerobot pattern)
        m = re.search(r"/([^/]+)/src/([^/]+)/", p)
        if m:
            pkg = m.group(2)
            abs_prefix = p[:m.end()]
            candidates[f"{pkg}/"].add(abs_prefix)
            continue
        # generic: take last 2 segments before file
        m = re.search(r"/([^/]+)/([^/]+)/[^/]+\.py$", p)
        if m:
            pkg = m.group(1)
            # parent dir of the file
            abs_prefix = p[:p.rfind(f"/{pkg}/") + len(f"/{pkg}/")]
            candidates[f"{pkg}/"].add(abs_prefix)

    # Resolve: for each pkg, pick the most common abs prefix
    result: dict[str, str] = {}
    for pkg, prefixes in candidates.items():
        # most common; if tie just take first
        ctr = Counter()
        for p in paths:
            for pre in prefixes:
                if p.startswith(pre):
                    ctr[pre] += 1
        if ctr:
            result[pkg] = ctr.most_common(1)[0][0]
    return result


# ============================================================================
# Step 2: parse parent class source to find self.<name> = <Class>(...) lines
# ============================================================================
def _find_assignment_lines(src_path: Path) -> dict[str, int]:
    """Parse a Python file, return {attribute_name: line_no} for `self.X = ...`
    assignments inside any __init__ method.

    These line numbers point to where modules are *declared* (rather than where
    forward calls them — which the trace records).
    """
    try:
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}

    out: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if (isinstance(tgt, ast.Attribute)
                            and isinstance(tgt.value, ast.Name)
                            and tgt.value.id == "self"):
                        # First assignment wins
                        out.setdefault(tgt.attr, stmt.lineno)
    return out


def infer_decl_src(trace: dict, path_map: dict[str, str]) -> dict[str, str]:
    """For each leaf module in module_tree, find its *declaration* line in the
    parent class's __init__, returning {node_id: 'pkg/path/file.py:line'}.

    node_id is the last segment of the module's dotted name (e.g. `q_proj`).
    For modules that share leaf names across the tree, we use the parent's
    qualname as a discriminator and emit one entry per unique leaf-name.

    The output here is keyed by the **last name segment** because that's
    typically the right granularity for educators ("all q_proj's point to the
    same Linear declaration in Llama's attention class"). For finer grain,
    consumers can supplement.
    """
    mt = trace.get("module_tree", {})

    # Build name → parent_qualname mapping
    by_name = {v["name"]: v for v in mt.values()}
    parent_of: dict[str, dict] = {}
    for v in mt.values():
        if v["name"] == "<root>" or "." not in v["name"]:
            continue
        parent_name = v["name"].rsplit(".", 1)[0]
        parent = by_name.get(parent_name)
        if parent:
            parent_of[v["name"]] = parent

    # For each (leaf_name_segment, parent_class) pair we encountered, find the
    # declaration line. We dedupe so we hit each parent file once per attr.
    seen: dict[tuple[str, str], int] = {}  # (parent_class_src, attr_name) -> lineno
    decl_src: dict[str, str] = {}

    def _rel_path(abs_path: str) -> str | None:
        for pkg, abs_prefix in path_map.items():
            if abs_path.startswith(abs_prefix):
                return pkg + abs_path[len(abs_prefix):]
        return None

    for full_name, node in by_name.items():
        if full_name == "<root>" or "." not in full_name:
            continue
        cls_name = node["qualname"].split(".")[-1]
        if cls_name not in LEAF_CLASSES:
            continue
        parent = parent_of.get(full_name)
        if not parent or not parent.get("src_file"):
            continue
        attr = full_name.rsplit(".", 1)[-1]
        # If attr is a number (.0, .1 for ModuleList), use the layer's name as attr.
        if attr.isdigit():
            # parent is the ModuleList — go up one more
            grandparent_name = parent["name"].rsplit(".", 1)[0] if "." in parent["name"] else None
            grandparent = by_name.get(grandparent_name) if grandparent_name else None
            if not grandparent or not grandparent.get("src_file"):
                continue
            attr = parent["name"].rsplit(".", 1)[-1]
            parent = grandparent
        parent_src = parent["src_file"]
        key = (parent_src, attr)
        if key in seen:
            lineno = seen[key]
        else:
            try:
                lines = _find_assignment_lines(Path(parent_src))
            except Exception:
                continue
            lineno = lines.get(attr)
            seen[key] = lineno
        if not lineno:
            continue
        rel = _rel_path(parent_src)
        if not rel:
            continue
        decl_src[attr] = f"{rel}:{lineno}"
    return decl_src


# ============================================================================
# Step 3: TREE skeleton — composite/leaf hierarchy with shape placeholders
# ============================================================================
def _collapse_repeated_layers(names: list[str]) -> tuple[list[str], dict[str, int]]:
    """Given a list of module dotted-names like ['enc.layers.0.attn', 'enc.layers.1.attn', ...],
    return the de-duplicated names (replacing .N. → .0.) and a {representative_name: repeat_count}
    map. Used to fold N identical transformer layers into a single visual node.
    """
    canonical: dict[str, set[int]] = defaultdict(set)
    for n in names:
        m = re.search(r"\.layers\.(\d+)\.", n)
        if m:
            idx = int(m.group(1))
            canon = re.sub(r"\.layers\.\d+\.", ".layers.0.", n, count=1)
            canonical[canon].add(idx)
        else:
            canonical[n].add(-1)
    deduped = list(canonical.keys())
    repeats = {k: len(v) for k, v in canonical.items() if -1 not in v and len(v) > 1}
    return deduped, repeats


def _make_node_id(name: str, strip_prefix: str = "") -> str:
    """Convert a dotted module name to a TREE node id.

    Strategy:
    - Optionally strip a common prefix shared by all nodes (e.g. 'model.')
    - Take the last 2 non-numeric segments (skip `0` from `.layers.0.` etc)
    - Replace dots with `_`
    - Strip leading digits (which would make invalid identifiers)
    Caller is expected to dedupe in case of collisions.
    """
    if strip_prefix and name.startswith(strip_prefix):
        name = name[len(strip_prefix):]
    parts = [p for p in name.split(".") if not p.isdigit()]
    if len(parts) >= 2:
        tail = "_".join(parts[-2:])
    elif parts:
        tail = parts[-1]
    else:
        tail = name.replace(".", "_")
    tail = re.sub(r"^layers_", "", tail)
    return re.sub(r"^\d+_?", "", tail) or "node"


def _common_prefix(names: list[str]) -> str:
    """Return the longest dotted-name prefix shared by all input names, with
    a trailing dot. Returns '' if no shared prefix (or all names are top-level).

    Used to strip vacuous wrapper segments like 'model.' from generated node ids
    when <root> has only one child class.
    """
    if not names:
        return ""
    split = [n.split(".") for n in names]
    common: list[str] = []
    for segs in zip(*split):
        if len(set(segs)) == 1:
            common.append(segs[0])
        else:
            break
    if not common:
        return ""
    # Always leave at least 1 segment in the resulting name, so strip only N-1
    if all(len(s) <= len(common) for s in split):
        common = common[:-1]
    return ".".join(common) + "." if common else ""


def _infer_hint(shape: list[int] | None) -> str:
    """Best-effort dimension-name hint from raw shape; Claude can override."""
    if not shape:
        return ""
    n = len(shape)
    if n == 4:
        return "image"
    if n == 3:
        return "seq"
    if n == 2:
        return ""
    return ""


def infer_tree_skeleton(trace: dict, decl_src: dict[str, str], path_map: dict[str, str]) -> dict[str, Any]:
    """Build a TREE dict scaffold from the trace's module_tree + first batch.

    Output structure follows render/tree_smolvla.py conventions:
      - 'pipeline': linear stages with all top-level direct children of <root>
        (Claude restructures into Y-shape later)
      - composite nodes for each parent module
      - leaf nodes for each LEAF_CLASSES instance
      - .layers.N collapsed to single leaf with repeat=N

    Semantic fields (name/sub/tooltip/what/blurb/formula/callout) are left
    as TODO placeholders — Claude fills these in.
    """
    mt = trace.get("module_tree", {})
    calls_by_name = pair_events_by_name(trace)

    # ---- Index modules by dotted name ----
    by_name = {v["name"]: v for v in mt.values()}
    children_of: dict[str, list[str]] = defaultdict(list)
    for v in mt.values():
        name = v["name"]
        if name == "<root>" or "." not in name:
            continue
        parent = name.rsplit(".", 1)[0]
        children_of[parent].append(name)

    # ---- Pick canonical (collapsed) name set ----
    all_names = [v["name"] for v in mt.values() if v["name"] != "<root>"]
    deduped, repeats = _collapse_repeated_layers(all_names)
    keep_set = set(deduped)

    # ---- Helpers ----
    def _rel_src(abs_path: str | None, lineno: Any = None) -> str:
        if not abs_path:
            return "—"
        for pkg, abs_prefix in path_map.items():
            if abs_path.startswith(abs_prefix):
                rel = pkg + abs_path[len(abs_prefix):]
                return f"{rel}:{lineno}" if lineno else rel
        return f"{abs_path}:{lineno}" if lineno else abs_path

    def _shape_summary(name: str) -> tuple[list[int] | None, list[int] | None, str, str]:
        calls = calls_by_name.get(name, [])
        if not calls:
            # try the .layers.0. equivalent
            for alt_name, cs in calls_by_name.items():
                if alt_name.replace(re.search(r"\.layers\.\d+\.", alt_name).group(0)
                                    if re.search(r"\.layers\.\d+\.", alt_name) else "",
                                    ".layers.0.") == name:
                    calls = cs
                    break
        if not calls:
            return None, None, "", ""
        first = calls[0]
        return (
            find_tensor_shape(first["inputs"]),
            find_tensor_shape(first["outputs"]),
            find_tensor_dtype(first["inputs"]) or "",
            find_tensor_dtype(first["outputs"]) or "",
        )

    # ---- Build node id → tree entry ----
    tree: dict[str, dict[str, Any]] = {}

    # Track which dotted names we've emitted, and what id we gave them.
    name_to_id: dict[str, str] = {}
    used_ids: set[str] = set()

    # Compute the common prefix (e.g. 'model.') to strip from all generated
    # node ids — keeps them concise when <root> has only one wrapper child.
    common_prefix = _common_prefix(all_names)

    def _assign_id(dotted: str) -> str:
        candidate = _make_node_id(dotted, strip_prefix=common_prefix)
        if candidate in used_ids:
            # disambiguate with longer suffix — skip numeric (layer-index) segments
            tail = dotted[len(common_prefix):] if dotted.startswith(common_prefix) else dotted
            parts = [p for p in tail.split(".") if not p.isdigit()]
            for take in range(3, len(parts) + 1):
                candidate = "_".join(parts[-take:]).replace(".", "_")
                if candidate not in used_ids:
                    break
        used_ids.add(candidate)
        name_to_id[dotted] = candidate
        return candidate

    # Top-level direct children of <root>
    root_qualname = next((v["qualname"] for v in mt.values() if v["name"] == "<root>"), "")
    top_level = sorted(
        [n for n in by_name if "." not in n and n != "<root>"],
        key=lambda n: n,
    )
    # If <root> has no dotted-level children, fall back to "model" depth-1
    if not top_level:
        top_level = sorted({n.split(".")[0] for n in all_names})
    # Auto-descend: if <root> has only one child (typical wrapper like
    # SmolVLAPolicy → VLAFlowMatching), use that child's children as top-level
    # so the user sees a meaningful Y-shape instead of one giant box.
    while len(top_level) == 1 and children_of.get(top_level[0]):
        descend_target = top_level[0]
        next_level = sorted(children_of[descend_target])
        # Only descend if next level has > 1 nodes (real branching)
        if len(next_level) > 1:
            top_level = next_level
            break
        top_level = next_level

    # Pre-assign IDs for top-level
    for n in top_level:
        _assign_id(n)

    def _emit_node(dotted: str):
        """Emit a tree entry for `dotted`. Recursive for composites."""
        if dotted not in keep_set and dotted not in by_name:
            return
        node = by_name[dotted]
        nid = name_to_id.get(dotted) or _assign_id(dotted)
        if nid in tree:
            return
        cls_name = node["qualname"].split(".")[-1]
        is_leaf = (cls_name in LEAF_CLASSES) or not children_of.get(dotted)

        if is_leaf:
            in_shape, out_shape, in_dtype, out_dtype = _shape_summary(dotted)
            attr_name = dotted.rsplit(".", 1)[-1] if "." in dotted else dotted
            src_str = decl_src.get(attr_name) or _rel_src(node.get("src_file"), node.get("src_line"))
            tree[nid] = {
                "type": "leaf",
                "name": f"TODO: {attr_name}",  # Claude fills concept_name
                "sub": f"{cls_name}",
                "tooltip": "TODO: 一行 hover 描述",
                "what": "TODO: 详细解释这个模块在做什么",
                "trace_name": dotted,
                "in_hint": _infer_hint(in_shape),
                "out_hint": _infer_hint(out_shape),
                "src": src_str,
            }
            if dotted in repeats:
                tree[nid]["repeat"] = repeats[dotted]
        else:
            kids = sorted(children_of[dotted])
            kids_kept = [k for k in kids if k in keep_set]
            kid_ids = []
            for k in kids_kept:
                _assign_id(k)
                _emit_node(k)
                kid_ids.append(name_to_id[k])
            tree[nid] = {
                "type": "composite",
                "name": f"TODO: {dotted.rsplit('.', 1)[-1] if '.' in dotted else dotted}",
                "sub": cls_name,
                "tooltip": "TODO: 一行 hover 描述",
                "blurb": "TODO: 这个 composite 的整体作用",
                "children": kid_ids,
            }
            if dotted in repeats:
                tree[nid]["repeat"] = repeats[dotted]

    for n in top_level:
        _emit_node(n)

    # Linear stages by default — Claude restructures into Y-shape after reading paper
    tree["pipeline"] = {
        "type": "composite",
        "name": "TODO: 整个 pipeline 的概念名",
        "sub": root_qualname.split(".")[-1] if root_qualname else "ROOT",
        "tooltip": "",
        "blurb": "",
        "layout": "stages",
        "story_line": "TODO: 一句话 framing — '模型不直接预测 X, 而是 Y...'",
        "stages": [{"nodes": [name_to_id[n]]} for n in top_level if n in name_to_id],
        "children": [name_to_id[n] for n in top_level if n in name_to_id],
    }
    return tree


# ============================================================================
# Step 4: digest.md — markdown summary for Claude to read
# ============================================================================
def build_digest(trace: dict, tree: dict, path_map: dict, decl_src: dict) -> str:
    mt = trace.get("module_tree", {})
    n_modules = len(mt)
    n_fwd = sum(1 for e in trace.get("module_events", []) if e["phase"] == "forward_post")
    n_bwd = sum(1 for e in trace.get("module_events", []) if e["phase"] == "backward")
    n_sdpa = len(trace.get("functional_events", []))

    root = next((v for v in mt.values() if v["name"] == "<root>"), {})
    root_qualname = root.get("qualname", "?")
    total_params = root.get("num_params_total", 0)

    leaf_classes = Counter()
    for v in mt.values():
        cls = v["qualname"].split(".")[-1]
        if cls in LEAF_CLASSES:
            leaf_classes[cls] += 1

    # Top-level children of <root>
    top_children = sorted(n for n in (v["name"] for v in mt.values()) if "." not in n and n != "<root>")

    lines: list[str] = []
    lines.append("# Trace 推断摘要")
    lines.append("")
    lines.append("> 这份是 `infer_tree.py` 从 trace JSON 机械化产出的摘要 + 骨架。")
    lines.append("> Claude 读这份 + 论文/README, 把 `inferred/tree_skeleton.py` 改成最终 TREE。")
    lines.append("")
    lines.append("## 一、trace 基本信息")
    lines.append("")
    lines.append(f"- 顶层 class: `{root_qualname}`")
    lines.append(f"- 总参数: {total_params/1e6:.1f}M")
    lines.append(f"- forward 调用数: {n_fwd}")
    lines.append(f"- backward 梯度数: {n_bwd}")
    lines.append(f"- functional ops (SDPA 等): {n_sdpa}")
    lines.append(f"- 模块树节点数: {n_modules}")
    lines.append("")
    lines.append("## 二、顶层结构（<root> 的直接子模块）")
    lines.append("")
    lines.append("这些是顶层 Y-shape 候选。请去 **顶层 class 的 forward()** 源码核对它们的"
                 "**语义顺序**（trace 顺序 ≠ forward 语法顺序，见 topology-inference.md Step 1）。")
    lines.append("")
    by_name = {v["name"]: v for v in mt.values()}
    for name in top_children:
        v = by_name[name]
        cls = v["qualname"].split(".")[-1]
        params = v.get("num_params_total", 0)
        lines.append(f"- `{name}` — {cls} ({params/1e6:.1f}M params)")
    lines.append("")
    lines.append("## 三、Leaf 类型统计")
    lines.append("")
    for cls, n in leaf_classes.most_common():
        lines.append(f"- {cls}: {n}")
    lines.append("")
    lines.append("## 四、识别到的 PATH_MAP")
    lines.append("")
    lines.append("```python")
    lines.append("PATH_MAP = {")
    for pkg, abs_p in sorted(path_map.items()):
        lines.append(f"    {pkg!r}: {abs_p!r},")
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("## 五、识别到的 DECL_SRC（声明位置）")
    lines.append("")
    lines.append(f"共 {len(decl_src)} 个 leaf 模块声明位置已定位（替换 trace 抓到的 torch 内部 forward 行）。")
    lines.append("")
    lines.append("```python")
    lines.append("DECL_SRC = {")
    for attr, src in sorted(decl_src.items()):
        lines.append(f"    {attr!r:30s}: {src!r},")
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("## 六、推断的 TREE 骨架")
    lines.append("")
    lines.append("已写到 `tree_skeleton.py`。其中:")
    lines.append("")
    lines.append("- 所有 `name` / `sub` / `tooltip` / `what` / `blurb` / `story_line` 都标了 `TODO:` —— **Claude 必须替换**")
    lines.append("- `trace_name` / `src` / `repeat` / `in_hint` / `out_hint` 已自动填好，**不要动**（除非真的错了）")
    lines.append("- `pipeline.stages` 默认是 linear（每个 stage 一个节点），**Claude 需要按论文重构成 Y-shape**")
    lines.append("- 没有 `is_heart` / `training_only` 标记 —— Claude 按 topology-inference.md Step 7 标")
    lines.append("- 没有 `formula` / `callout` —— Claude 按 explanation-style.md 补")
    lines.append("")
    lines.append("## 七、下一步（给 Claude 的 checklist）")
    lines.append("")
    lines.append("> **⚠️ 准确性 > 速度**。不要试图把这个模型快速归类到某个常见拓扑（Y-shape / U-Net / dual-encoder ...）—— 那是锚定陷阱。")
    lines.append("> 唯一可靠的方法是**亲自读 `<root>` class 的 forward 源码 + 论文 abstract**，从代码真实的依赖关系推拓扑。")
    lines.append("> 罕见架构（mamba / MoE / RWKV / 混合）多花 10 分钟读源码，远比 30 秒套模板套错值。")
    lines.append("")
    lines.append("1. **打开顶层 forward 源码** — `<root>` class（上面第一节给了 qualname + src_file），逐行读 forward 方法")
    lines.append("2. **抓 story_line** — 论文 abstract 末两句通常就是；找不到就读 forward 末尾的 loss 反推 training objective")
    lines.append("3. **按 forward 语法顺序** 列出所有 `self.X(...)` 调用 + 关键变量赋值，画依赖图（哪个变量来自哪些）—— 这是 pipeline.stages 的真实拓扑")
    lines.append("4. **给每个顶层 node 起 concept_name + purpose_line**（Step 3-4 of topology-inference.md）")
    lines.append("5. **标心脏方块 (is_heart) + 训练 only (training_only)**（Step 7 of topology-inference.md）")
    lines.append("6. **给关键 leaf 补 formula / callout / what**（Step 5 of topology-inference.md）—— 文字必须源自论文 / 注释 / 源码 docstring, **不要 LLM 凭空想象**")
    lines.append("7. **把改好的 tree_skeleton.py 重命名为 `render/tree_<repo>.py`**, 再在 `render/__init__.py` 换 TREE 的 import 来源")
    lines.append("8. **跑一遍 + 走 verify checklist**（`references/topology-inference.md` 末尾）—— 不通过就回上一步重做, 不要让“差不多就行”的产物上线")
    return "\n".join(lines)


# ============================================================================
# Output writers
# ============================================================================
def _write_python_module(path: Path, module_doc: str, var_name: str, value: Any) -> None:
    """Write a Python module file with a single dict/value assignment."""
    body = json.dumps(value, ensure_ascii=False, indent=4)
    # Convert JSON to Python-style (true → True, null → None, false → False)
    body = body.replace("true", "True").replace("false", "False").replace("null", "None")
    content = f'"""{module_doc}"""\n\n{var_name} = {body}\n'
    path.write_text(content, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace_json", type=Path, help="path to trace_<ts>.json")
    ap.add_argument("--out-dir", type=Path, default=Path("inferred"),
                    help="output directory (default: ./inferred)")
    args = ap.parse_args()

    trace = json.loads(args.trace_json.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[infer] loading {args.trace_json} ({args.trace_json.stat().st_size//1024} KB)")

    path_map = infer_path_map(trace)
    print(f"[infer] PATH_MAP: {len(path_map)} prefixes")

    decl_src = infer_decl_src(trace, path_map)
    print(f"[infer] DECL_SRC: {len(decl_src)} leaf declarations located")

    tree = infer_tree_skeleton(trace, decl_src, path_map)
    n_leaves = sum(1 for v in tree.values() if v.get("type") == "leaf")
    n_composites = sum(1 for v in tree.values() if v.get("type") == "composite")
    print(f"[infer] TREE skeleton: {n_leaves} leaves + {n_composites} composites")

    digest = build_digest(trace, tree, path_map, decl_src)

    _write_python_module(
        args.out_dir / "path_map.py",
        "Auto-detected PATH_MAP (from infer_tree.py).",
        "PATH_MAP", path_map)
    _write_python_module(
        args.out_dir / "decl_src.py",
        "Auto-detected DECL_SRC (from infer_tree.py).\n\n"
        "Maps leaf attribute names → declaration location ('pkg/path/file.py:line').\n"
        "Replaces trace's torch-internal forward call line.",
        "DECL_SRC", decl_src)
    _write_python_module(
        args.out_dir / "tree_skeleton.py",
        "Auto-generated TREE skeleton (from infer_tree.py).\n\n"
        "Mechanical structure is correct; semantic fields are TODO placeholders.\n"
        "See digest.md for what Claude needs to fill in.",
        "TREE", tree)
    (args.out_dir / "digest.md").write_text(digest, encoding="utf-8")

    print(f"[infer] wrote 4 files to {args.out_dir}/")
    print(f"        → path_map.py")
    print(f"        → decl_src.py")
    print(f"        → tree_skeleton.py")
    print(f"        → digest.md  (← Claude reads this first)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
