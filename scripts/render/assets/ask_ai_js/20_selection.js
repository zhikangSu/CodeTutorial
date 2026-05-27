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
