  // ---- Wire up event listeners ----
  function init() {
    S.config = loadConfig();
    updateModelTag();
    setSendButtonMode('send');  // 确保 dataset.mode = 'send' 初始值

    // Selection -> show fab (统一走 recheckSelection, debounce 100ms)
    // selectionchange 是主控 (任何方式的选区变化都触发), mouseup/keyup 兜底
    document.addEventListener('selectionchange', recheckSelection);
    document.addEventListener('mouseup', recheckSelection);
    document.addEventListener('keyup', e => {
      // Shift+arrow 等键盘选区
      if (e.shiftKey || e.key === 'ArrowLeft' || e.key === 'ArrowRight'
          || e.key === 'ArrowUp' || e.key === 'ArrowDown') recheckSelection();
    });
    document.addEventListener('scroll', hideFab, true);

    // 兜底: 点击任何地方 (非 FAB / 非 panel) 后立刻同步检查选区状态
    // — 解决"点空白处取消选中, 但 FAB 还在"的 bug, 不等 selectionchange debounce
    document.addEventListener('click', e => {
      if (e.target.closest && (e.target.closest('#ask-ai-fab')
                            || e.target.closest('.ask-ai-panel')
                            || e.target.closest('.ask-ai-config'))) return;
      // setTimeout 0 让浏览器先处理 click → selection collapse
      setTimeout(() => {
        const sel = window.getSelection();
        if (!sel || sel.isCollapsed || sel.toString().trim().length < 5) {
          hideFab();
          S.selection = null;
        }
      }, 0);
    });

    const fab = document.getElementById('ask-ai-fab');
    fab.addEventListener('mousedown', e => e.preventDefault());  // don't lose selection
    fab.addEventListener('click', () => { hideFab(); openPanel({}); });

    // Panel controls
    document.getElementById('ask-ai-close').addEventListener('click', closePanel);
    document.getElementById('ask-ai-reset').addEventListener('click', resetConversation);
    document.getElementById('ask-ai-settings').addEventListener('click', () => openConfig(false));

    // Submit (按钮在 stop 模式时点击 = 中止流式; 否则 = 发送)
    document.getElementById('ask-ai-form').addEventListener('submit', e => {
      e.preventDefault();
      const btn = document.getElementById('ask-ai-send');
      if (btn && btn.dataset.mode === 'stop' && S.abortCtrl) {
        S.abortCtrl.abort();
        return;
      }
      const input = document.getElementById('ask-ai-input');
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      sendMessage(text);
    });
    document.getElementById('ask-ai-input').addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        document.getElementById('ask-ai-form').dispatchEvent(new Event('submit', {cancelable: true}));
      }
    });

    // Config modal
    document.getElementById('ask-ai-config-close').addEventListener('click', closeConfig);
    document.getElementById('ask-ai-config-backdrop').addEventListener('click', closeConfig);
    document.getElementById('cfg-provider').addEventListener('change', e => applyProviderPreset(e.target.value));
    document.getElementById('cfg-model-refresh').addEventListener('click', refreshModels);
    document.getElementById('cfg-test').addEventListener('click', testConfig);
    document.getElementById('cfg-save').addEventListener('click', () => {
      const cfg = collectConfig();
      if (!cfg.baseUrl || !cfg.model) {
        const status = document.getElementById('cfg-status');
        status.textContent = '✗ Base URL 和 Model 必填';
        status.className = 'ask-ai-config-status err';
        return;
      }
      saveConfig(cfg);
      closeConfig();
      flashToast('API 配置已保存');
    });

    // In-panel model quickswitch
    document.getElementById('ask-ai-model-tag').addEventListener('click', toggleModelDropdown);
    // 点击 dropdown 外部 → 关闭
    document.addEventListener('click', e => {
      const switcher = document.getElementById('model-switcher');
      if (switcher && !switcher.contains(e.target)) {
        const dd = document.getElementById('model-dropdown');
        if (dd && !dd.hidden) dd.hidden = true;
      }
    });
  }
