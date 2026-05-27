  // ---- Config modal ----
  function openConfig(firstTime) {
    S.configOpen = true;
    document.getElementById('ask-ai-config-backdrop').hidden = false;
    const modal = document.getElementById('ask-ai-config');
    modal.hidden = false;
    // populate
    const cfg = S.config || {};
    const defaults = PROVIDER_DEFAULTS[cfg.provider || 'gemini'] || PROVIDER_DEFAULTS.gemini;
    document.getElementById('cfg-provider').value = cfg.provider || 'gemini';
    document.getElementById('cfg-base-url').value = cfg.baseUrl || defaults.base;
    document.getElementById('cfg-api-key').value = cfg.apiKey || '';
    // model: 优先用缓存的 list 填 select, 否则手填 input
    const cached = cfg.cachedModels || [];
    const sel = document.getElementById('cfg-model-select');
    const inp = document.getElementById('cfg-model-input');
    if (cached.length) {
      sel.innerHTML = cached.map(m =>
        '<option value="' + escapeHtml(m) + '">' + escapeHtml(m) + '</option>'
      ).join('');
      if (cfg.model && cached.includes(cfg.model)) sel.value = cfg.model;
      sel.hidden = false; inp.hidden = true;
      document.getElementById('cfg-model-hint').textContent = '使用上次拉到的 ' + cached.length + ' 个 model (↻ 重新拉取)';
      document.getElementById('cfg-model-hint').className = 'ask-ai-field-hint ok';
    } else {
      sel.hidden = true; inp.hidden = false;
      inp.value = cfg.model || defaults.model;
      document.getElementById('cfg-model-hint').textContent = '↻ 点刷新可从 /v1/models 拉取列表';
      document.getElementById('cfg-model-hint').className = 'ask-ai-field-hint';
    }
    // inference 参数
    document.getElementById('cfg-temp').value      = cfg.temperature != null ? cfg.temperature : 0.4;
    document.getElementById('cfg-max-tokens').value = cfg.maxTokens != null ? cfg.maxTokens : 8192;
    document.getElementById('cfg-top-p').value     = cfg.topP != null ? cfg.topP : 1.0;
    document.getElementById('cfg-retention').value = cfg.retention != null ? cfg.retention : 5;
    document.getElementById('cfg-sys-prompt').value = cfg.systemPrompt || '';

    document.getElementById('cfg-status').textContent = firstTime ? '👋 第一次使用 — 选 provider 后保存即可 (高级参数留默认就行)' : '';
    document.getElementById('cfg-status').className = 'ask-ai-config-status';
  }
  function closeConfig() {
    S.configOpen = false;
    document.getElementById('ask-ai-config-backdrop').hidden = true;
    document.getElementById('ask-ai-config').hidden = true;
  }
  function applyProviderPreset(provider) {
    const preset = PROVIDER_DEFAULTS[provider] || PROVIDER_DEFAULTS.custom;
    document.getElementById('cfg-base-url').value = preset.base;
    document.getElementById('cfg-model-input').value = preset.model;
    // 切 provider 时 model 列表失效, 回到手填
    document.getElementById('cfg-model-select').hidden = true;
    document.getElementById('cfg-model-input').hidden = false;
    document.getElementById('cfg-model-hint').textContent = '↻ 点刷新可从该 provider 拉取 model 列表';
    document.getElementById('cfg-model-hint').className = 'ask-ai-field-hint';
  }

  function collectConfig() {
    const sel = document.getElementById('cfg-model-select');
    const inp = document.getElementById('cfg-model-input');
    const modelFromSelect = !sel.hidden && sel.value;
    const modelFromInput = inp.value.trim();
    return {
      provider: document.getElementById('cfg-provider').value,
      baseUrl:  document.getElementById('cfg-base-url').value.trim(),
      apiKey:   document.getElementById('cfg-api-key').value.trim(),
      model:    modelFromSelect || modelFromInput,
      temperature: parseFloat(document.getElementById('cfg-temp').value) || 0.4,
      maxTokens:   parseInt(document.getElementById('cfg-max-tokens').value, 10) || 8192,
      topP:        parseFloat(document.getElementById('cfg-top-p').value) || 1.0,
      retention:   parseInt(document.getElementById('cfg-retention').value, 10),
      systemPrompt: document.getElementById('cfg-sys-prompt').value,
      // 缓存最近一次拉到的 model list, 给 in-panel 快切器用
      cachedModels: S.config && S.config.cachedModels || [],
    };
  }

  // ---- /v1/models 拉取 ----
  async function fetchModelList() {
    const cfg = collectConfig();
    if (!cfg.baseUrl) return { error: '先填 Base URL' };
    const url = cfg.baseUrl.replace(/\/+$/, '') + '/models';
    try {
      const headers = {};
      if (cfg.apiKey) headers['Authorization'] = 'Bearer ' + cfg.apiKey;
      const resp = await fetch(url, { headers });
      if (!resp.ok) {
        if (resp.status === 401) return { error: '401 (API key 无效)' };
        if (resp.status === 403) return { error: '403 (拒绝访问)' };
        if (resp.status === 404) return { error: '404 (endpoint 不支持 /models)' };
        return { error: 'HTTP ' + resp.status };
      }
      const data = await resp.json();
      let models = [];
      if (Array.isArray(data)) models = data.map(m => typeof m === 'string' ? m : (m.id || m.name));
      else if (data.data && Array.isArray(data.data)) models = data.data.map(m => m.id || m.name);
      else if (data.models && Array.isArray(data.models)) models = data.models.map(m => typeof m === 'string' ? m : (m.id || m.name || m.model));
      models = models.filter(Boolean);
      if (!models.length) return { error: 'endpoint 返回空列表' };
      return { models };
    } catch (e) {
      const msg = e.message || String(e);
      const cors = msg.includes('Failed to fetch') || msg.includes('NetworkError') ? ' (可能 CORS)' : '';
      return { error: msg + cors };
    }
  }

  async function refreshModels() {
    const btn = document.getElementById('cfg-model-refresh');
    const hint = document.getElementById('cfg-model-hint');
    const sel = document.getElementById('cfg-model-select');
    const inp = document.getElementById('cfg-model-input');
    btn.classList.add('loading');
    hint.textContent = '拉取中...';
    hint.className = 'ask-ai-field-hint';
    const result = await fetchModelList();
    btn.classList.remove('loading');
    if (result.error) {
      sel.hidden = true; inp.hidden = false;
      hint.textContent = '拉取失败: ' + result.error + ' — 已回退手填模式';
      hint.className = 'ask-ai-field-hint err';
      return;
    }
    // populate dropdown
    sel.innerHTML = result.models.map(m =>
      '<option value="' + escapeHtml(m) + '">' + escapeHtml(m) + '</option>'
    ).join('');
    // 保留当前选中的 model 如果在列表里
    const currentModel = (S.config && S.config.model) || inp.value.trim();
    if (currentModel && result.models.includes(currentModel)) sel.value = currentModel;
    sel.hidden = false; inp.hidden = true;
    hint.textContent = '✓ 已拉到 ' + result.models.length + ' 个 model';
    hint.className = 'ask-ai-field-hint ok';
    // 缓存给 in-panel 快切器
    if (S.config) S.config.cachedModels = result.models;
  }
  async function testConfig() {
    const cfg = collectConfig();
    const status = document.getElementById('cfg-status');
    status.textContent = '测试中...';
    status.className = 'ask-ai-config-status';
    try {
      const url = cfg.baseUrl.replace(/\/+$/, '') + '/chat/completions';
      const resp = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + cfg.apiKey,
        },
        body: JSON.stringify({
          model: cfg.model,
          messages: [{role: 'user', content: 'hi'}],
          max_tokens: 2,
          stream: false,
        }),
      });
      if (resp.ok) {
        status.textContent = '✓ 连接成功';
        status.className = 'ask-ai-config-status ok';
      } else {
        const text = await resp.text();
        status.textContent = '✗ HTTP ' + resp.status + ': ' + text.slice(0, 120);
        status.className = 'ask-ai-config-status err';
      }
    } catch (e) {
      status.textContent = '✗ ' + e.message + (String(e).includes('CORS') || String(e).includes('Failed to fetch')
        ? ' (可能是 CORS 问题 — 试着用 python -m http.server 起本地 server 再访问)' : '');
      status.className = 'ask-ai-config-status err';
    }
  }
