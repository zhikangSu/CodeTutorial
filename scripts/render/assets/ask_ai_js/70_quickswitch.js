  // ---- In-panel model quickswitch ----
  function toggleModelDropdown(e) {
    if (e) e.stopPropagation();
    const dd = document.getElementById('model-dropdown');
    if (!dd.hidden) { dd.hidden = true; return; }
    if (!S.config) { openConfig(true); return; }
    const models = (S.config.cachedModels && S.config.cachedModels.length)
                   ? S.config.cachedModels
                   : (S.config.model ? [S.config.model] : []);
    if (!models.length) {
      dd.innerHTML = '<div class="ask-model-dropdown-empty">无 model 列表 — 去 ⚙ 设置里点 ↻ 拉取</div>';
    } else {
      dd.innerHTML = models.map(m =>
        '<button type="button" data-model="' + escapeHtml(m) + '"' + (m === S.config.model ? ' class="current"' : '') + '>'
        + escapeHtml(m) + '</button>'
      ).join('');
      dd.querySelectorAll('button[data-model]').forEach(btn => {
        btn.addEventListener('click', () => attemptModelSwitch(btn.dataset.model));
      });
    }
    dd.hidden = false;
  }

  function attemptModelSwitch(newModel) {
    const dd = document.getElementById('model-dropdown');
    dd.hidden = true;
    if (!S.config || newModel === S.config.model) return;
    if (S.messages.length === 0) {
      // 没历史, 直接切
      _applyModelSwitch(newModel, false);
      return;
    }
    // 有历史 — 弹小确认条问是否清空
    const switcher = document.getElementById('model-switcher');
    const confirm = document.createElement('div');
    confirm.className = 'ask-switch-confirm';
    confirm.innerHTML =
      '切换到 <strong>' + escapeHtml(newModel) + '</strong> 后, 之前的对话上下文在新模型里可能表现异常。'
      + '<div class="ask-switch-confirm-actions">'
      + '  <button type="button" class="primary" data-action="clear">清空 + 切换</button>'
      + '  <button type="button" data-action="keep">保留对话</button>'
      + '  <button type="button" data-action="cancel">取消</button>'
      + '</div>';
    switcher.appendChild(confirm);
    confirm.querySelectorAll('button').forEach(b => {
      b.addEventListener('click', () => {
        const a = b.dataset.action;
        confirm.remove();
        if (a === 'cancel') return;
        if (a === 'clear') resetConversation();
        _applyModelSwitch(newModel, a === 'keep');
      });
    });
  }

  function _applyModelSwitch(newModel, keepHistory) {
    S.config.model = newModel;
    saveConfig(S.config);
    if (keepHistory) {
      flashToast('已切到 ' + newModel + ' (对话历史保留, 注意新模型可能不一致)', false);
    } else {
      flashToast('已切到 ' + newModel);
    }
  }

  // ---- Send button mode (send / stop) ----
  function setSendButtonMode(mode) {
    const btn = document.getElementById('ask-ai-send');
    if (!btn) return;
    btn.dataset.mode = mode;
    if (mode === 'stop') {
      btn.innerHTML = '<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><rect x="4" y="4" width="8" height="8" rx="1.2"/></svg>';
      btn.title = '停止生成';
      btn.classList.add('stop-mode');
    } else {
      btn.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 8 L14 2 L11 14 L8 9 L2 8 Z"/></svg>';
      btn.title = '发送 (Ctrl+Enter)';
      btn.classList.remove('stop-mode');
    }
  }
