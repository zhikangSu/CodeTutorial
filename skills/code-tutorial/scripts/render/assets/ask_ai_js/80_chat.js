  // ---- Send message (streaming) ----
  async function sendMessage(userText) {
    if (S.isStreaming) return;
    if (!userText || !userText.trim()) return;
    if (!S.config) { openConfig(true); return; }

    // ---- 选中文本以 markdown 引用形式拼进 user 消息 (用户可见 + LLM 一定看得到) ----
    // 这比 context 注入更可靠 — context 是 metadata 容易被 LLM 当背景忽略,
    // 直接塞进 user message 的引用块是显式的"我在问这段"信号。
    let displayUserText = userText;
    let payloadUserText = userText;
    if (S.pinnedSelection) {
      const quoted = '> ' + S.pinnedSelection.split('\n').join('\n> ');
      displayUserText = quoted + '\n\n' + userText;
      payloadUserText = displayUserText;  // 发给 LLM 的也是引用 + 问题
      S.pinnedSelection = null;  // 一次性引用, 后续轮不再重复 (上下文已建立)
    }

    // ---- Compact context: 第一轮发完整, 之后只在 context 变化时重发 ----
    const isFirst = S.messages.length === 0;
    const fullCtx = snapshotContext({moduleId: state.drawerOpen});
    // djb2 hash
    let h = 5381;
    for (let i = 0; i < fullCtx.length; i++) h = ((h << 5) + h) + fullCtx.charCodeAt(i);
    const ctxHash = String(h);
    let fullUserContent;
    if (isFirst || ctxHash !== S.lastSentContextHash) {
      fullUserContent = fullCtx + '\n## 用户的问题\n' + payloadUserText;
      S.lastSentContextHash = ctxHash;
    } else {
      // 后续轮且 context 未变: 只发问题, 节省 token
      fullUserContent = payloadUserText;
    }

    S.messages.push({role: 'user', content: fullUserContent});

    appendMessage('user', displayUserText);  // UI 显示引用 + 问题, 让用户看见自己发了啥
    const assistantEl = appendMessage('assistant', '');
    assistantEl.classList.add('streaming');
    // 准备 thinking 区 + content 区 (即使 provider 不返回 thinking, 也无害)
    const bodyEl = assistantEl.querySelector('.ask-msg-body');
    bodyEl.innerHTML =
        '<details class="ask-thinking" id="" hidden><summary>思考过程</summary><div class="ask-thinking-content"></div></details>'
      + '<div class="ask-content"></div>';
    const thinkingEl = bodyEl.querySelector('.ask-thinking');
    const thinkingContentEl = bodyEl.querySelector('.ask-thinking-content');
    const contentEl = bodyEl.querySelector('.ask-content');

    // ---- 对话保留轮数 (retention) — 仅截 history, 当前轮一定要送 ----
    let convoMessages = S.messages;
    const retention = S.config.retention != null ? S.config.retention : 5;
    if (retention > 0) {
      // 一"轮" = user + assistant 各一条, 所以保留 retention*2 条 + 当前 user 这条
      // 取 messages 末尾的 retention*2 + 1 条 (当前 user 永远在末尾)
      const keep = retention * 2 + 1;
      if (S.messages.length > keep) convoMessages = S.messages.slice(-keep);
    }

    const url = S.config.baseUrl.replace(/\/+$/, '') + '/chat/completions';
    const payload = {
      model: S.config.model,
      messages: [
        {role: 'system', content: buildSystemPrompt()},
        ...convoMessages,
      ],
      stream: true,
      temperature: S.config.temperature != null ? S.config.temperature : 0.4,
      top_p: S.config.topP != null ? S.config.topP : 1.0,
      max_tokens: S.config.maxTokens != null ? S.config.maxTokens : 8192,
    };

    S.isStreaming = true;
    S.abortCtrl = new AbortController();
    setSendButtonMode('stop');  // 切到"停止生成"按钮
    let fullText = '';
    let thinkingText = '';
    let lastKatexRender = 0;

    try {
      const resp = await fetch(url, {
        method: 'POST',
        signal: S.abortCtrl.signal,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + (S.config.apiKey || ''),
        },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const text = await resp.text();
        assistantEl.remove();
        appendError(diagnoseHttpError(resp.status, text, url), true);
        return;
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;  // SSE event separator (空行)
          // 跳过 SSE meta fields: retry: / id: / event: / : (comment)
          // —— 经过第三方网关或本地 proxy 时这些字段都可能出现, 不能无脑 JSON.parse
          if (trimmed.startsWith(':')) continue;
          if (trimmed.startsWith('retry:') || trimmed.startsWith('id:') || trimmed.startsWith('event:')) continue;
          if (!trimmed.startsWith('data:')) continue;
          const data = trimmed.slice(5).trim();
          if (!data || data === '[DONE]') continue;
          try {
            const parsed = JSON.parse(data);
            const choice = parsed.choices && parsed.choices[0];
            if (!choice) continue;
            const delta = choice.delta || {};
            // 思考过程: DeepSeek = reasoning_content, Anthropic = thinking, Gemini = thoughts (有时)
            const thinkChunk = delta.reasoning_content || delta.thinking || delta.thoughts;
            const contentChunk = delta.content;
            if (thinkChunk) {
              thinkingText += thinkChunk;
              thinkingEl.hidden = false;
              thinkingEl.open = true;  // 流式时展开
              thinkingContentEl.textContent = thinkingText;
            }
            if (contentChunk) {
              fullText += contentChunk;
              // 思考结束 + 正文开始 → 收起 thinking
              if (thinkingEl && !thinkingEl.dataset.collapsedOnce && thinkingText) {
                thinkingEl.open = false;
                thinkingEl.dataset.collapsedOnce = '1';
              }
              // 即时显示纯文本 (cheap)
              contentEl.textContent = fullText;
              const msgs = document.getElementById('ask-ai-messages');
              msgs.scrollTop = msgs.scrollHeight;
              // markdown + KaTeX 节流到 150ms 一次
              const now = performance.now();
              if (now - lastKatexRender > KATEX_THROTTLE_MS) {
                lastKatexRender = now;
                renderAssistantMarkdown(contentEl, fullText);
              }
            }
          } catch (e) { /* partial json across chunks, skip */ }
        }
      }
      // streaming done — 最终一次完整渲染 (确保所有公式 + 代码块 + delimiter 都闭合渲染)
      assistantEl.classList.remove('streaming');
      if (fullText) renderAssistantMarkdown(contentEl, fullText);
      // thinking 区流完后默认折叠 (用户想看可手动展开)
      if (thinkingText) thinkingEl.open = false;
      else thinkingEl.hidden = true;
      S.messages.push({role: 'assistant', content: fullText});
    } catch (e) {
      assistantEl.classList.remove('streaming');
      if (e.name === 'AbortError') {
        // 用户主动点了"停止" — 保留已收到的内容, 加个标记
        renderAssistantMarkdown(contentEl, (fullText || '') + '\n\n_[已停止生成]_');
        if (thinkingText) { thinkingEl.open = false; } else { thinkingEl.hidden = true; }
        if (fullText) S.messages.push({role: 'assistant', content: fullText});
      } else {
        assistantEl.remove();
        appendError(diagnoseFetchError(e, url), true);
      }
    } finally {
      S.isStreaming = false;
      S.abortCtrl = null;
      setSendButtonMode('send');  // 恢复发送按钮
    }
  }
