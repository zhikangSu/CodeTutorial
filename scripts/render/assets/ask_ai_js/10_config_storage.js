  // ---- Config storage ----
  function loadConfig() {
    try {
      const raw = localStorage.getItem(CONFIG_KEY);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch (e) { return null; }
  }
  function saveConfig(cfg) {
    localStorage.setItem(CONFIG_KEY, JSON.stringify(cfg));
    S.config = cfg;
    updateModelTag();
  }
  function updateModelTag() {
    const tag = document.getElementById('ask-ai-model-tag');
    if (!tag) return;
    if (S.config && S.config.model) {
      tag.textContent = S.config.model;
      tag.classList.remove('unconfigured');
    } else {
      tag.textContent = '未配置 (点 ⚙ 设置)';
      tag.classList.add('unconfigured');
    }
  }
