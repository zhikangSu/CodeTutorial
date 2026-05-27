"""render package — split-out modules for render_html.py.

Public API:
    load_trace, pair_events_by_name        (trace_utils)
    enrich_with_trace                      (enrich)
    TREE                                   (tree_smolvla)
    bundle_assets                          (assets)
"""

from .trace_utils import (
    load_trace,
    pair_events_by_name,
    find_tensor_shape,
    find_tensor_dtype,
    fmt_shape,
    fmt_arrow,
)
from .source_links import PATH_MAP, SMOLVLA_DECL_SRC, vscode_link, activation_svg
from .tree_smolvla import TREE
from .enrich import enrich_with_trace
from .assets import bundle_assets

__all__ = [
    "load_trace",
    "pair_events_by_name",
    "find_tensor_shape",
    "find_tensor_dtype",
    "fmt_shape",
    "fmt_arrow",
    "PATH_MAP",
    "SMOLVLA_DECL_SRC",
    "vscode_link",
    "activation_svg",
    "TREE",
    "enrich_with_trace",
    "bundle_assets",
]
