  // ---- 错误诊断: 把 HTTP 错误码 / fetch 异常 翻译成可操作的中文提示 ----
  function diagnoseHttpError(status, body, endpoint) {
    if (status === 401) return 'API key 无效或过期。请点上方 ⚙ 在设置里更新 key。';
    if (status === 403) return 'API 拒绝访问 (403)。可能 key 没启用对应 model, 或 region 受限。';
    if (status === 404) return 'Endpoint 不存在: ' + endpoint + '\n检查 Base URL 是否正确。';
    if (status === 429) return '触发限流 (429)。等几秒重试, 或换 model / 升级 plan。';
    if (status >= 500) return '服务器错误 ' + status + '。endpoint 暂时不可用, 稍后重试。';
    const trimmed = String(body || '').trim();
    return 'HTTP ' + status + (trimmed ? ': ' + trimmed.slice(0, 240) : '');
  }
  function diagnoseFetchError(err, endpoint) {
    const msg = String(err.message || err);
    const protocol = location.protocol;
    // 最高频的首次失败: file:// 撞 CORS
    if (protocol === 'file:' && (msg.includes('Failed to fetch') || msg.includes('NetworkError'))) {
      return '看起来你是双击 HTML 用 file:// 打开的, 浏览器禁止从 file:// 直接请求外部 API。\n'
           + '解决: 起一个本地 server, 例如\n'
           + '  cd <html 所在目录> && python -m http.server 8000\n'
           + '然后用 http://localhost:8000/<filename>.html 访问。';
    }
    if (msg.includes('Failed to fetch') || msg.includes('NetworkError')) {
      if (endpoint.includes('localhost') || endpoint.includes('127.0.0.1')) {
        return '本地 endpoint 不可达: ' + endpoint + '\n确认 Ollama / 本地 server 真的在跑 (例: ollama serve)。';
      }
      return '请求发不出: ' + msg + '\n常见原因: endpoint 拒绝浏览器直连 CORS / 网络不可达 / endpoint 输错。';
    }
    return msg;
  }

  // ---- 流式 KaTeX 节流: 期间 150ms 才重渲一次 markdown + 数学, 中间用 textContent ----
  const KATEX_THROTTLE_MS = 150;
