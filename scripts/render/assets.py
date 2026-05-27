"""Asset loader: read CSS / JS / HTML files from assets/ as strings.

ask_ai_js/ is concatenated in alphabetical order (00_intro → 99_outro)
to form a single IIFE — function declarations are hoisted to the top of
the IIFE scope, so cross-file function references work regardless of
which numbered file the caller is defined in.
"""

from __future__ import annotations

from pathlib import Path

_ASSETS = Path(__file__).parent / "assets"


def _read(rel: str) -> str:
    return (_ASSETS / rel).read_text(encoding="utf-8")


def _read_concat(subdir: str) -> str:
    files = sorted((_ASSETS / subdir).glob("*.js"))
    return "\n".join(p.read_text(encoding="utf-8") for p in files)


def bundle_assets() -> dict[str, str]:
    """Return all assets as a dict of {name: contents-string}.

    Returned keys:
        main_css, ask_ai_css      — joined into <style>
        katex_head                — goes inside <head>
        ask_ai_html               — goes inside <body>
        main_js                   — main tree render (contains __TREE_JSON__ placeholder)
        ask_ai_js                 — IIFE assembled from ask_ai_js/*.js
    """
    return {
        "main_css":     _read("main.css"),
        "ask_ai_css":   _read("ask_ai.css"),
        "katex_head":   _read("katex.html").rstrip("\n"),
        "ask_ai_html":  _read("ask_ai.html"),
        "main_js":      _read("main.js"),
        "ask_ai_js":    _read_concat("ask_ai_js"),
    }
