  // ---- Panel ----
  function openPanel(opts) {
    opts = opts || {};
    if (!S.config || !S.config.apiKey && S.config.provider !== 'ollama') {
      // First-time: open config first (but allow ollama with empty key)
      if (!S.config) { openConfig(true); return; }
    }
    // 把当前选区"锁死"成 pinnedSelection — 接下来 textarea 拿焦点导致
    // document.selection 被清空也不影响, 第一条 user message 会用 pinnedSelection。
    if (S.selection) S.pinnedSelection = S.selection;
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
    S.pinnedSelection = null;  // 关 panel 时清掉锁定的选区
    const panel = document.getElementById('ask-ai-panel');
    panel.classList.remove('open');
    panel.setAttribute('aria-hidden', 'true');
    if (S.abortCtrl) { S.abortCtrl.abort(); S.abortCtrl = null; }
    setTimeout(() => { panel.hidden = true; }, 260);
  }
  function resetConversation() {
    S.messages = [];
    S.lastSentContextHash = '';  // 重置 context cache, 下次发会重新带完整 context
    S.pinnedSelection = null;    // 新对话也清掉旧选区
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
    // 优先用锁定的选区, 否则用当前实时选区
    const selText = S.pinnedSelection || S.selection;
    if (selText) {
      const truncated = selText.length > 80 ? selText.slice(0, 80) + '…' : selText;
      html += '<div>选中 (将作为下一条问题的引用): <span class="ctx-sel">"' + escapeHtml(truncated) + '"</span></div>';
    }
    if (breadcrumb || moduleId || selText) {
      pill.innerHTML = html;
      pill.hidden = false;
    } else {
      pill.hidden = true;
    }
  }
