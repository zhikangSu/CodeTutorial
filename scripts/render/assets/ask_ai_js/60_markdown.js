  // ---- UI helpers ----
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function clearEmptyState() {
    const msgs = document.getElementById('ask-ai-messages');
    const empty = msgs.querySelector('.ask-ai-empty');
    if (empty) empty.remove();
  }
  function appendMessage(role, text) {
    clearEmptyState();
    const msgs = document.getElementById('ask-ai-messages');
    const wrap = document.createElement('div');
    wrap.className = 'ask-msg ' + role;
    wrap.innerHTML = '<div class="ask-msg-role">' + (role === 'user' ? 'YOU' : 'AI') + '</div>'
                   + '<div class="ask-msg-body"></div>';
    const body = wrap.querySelector('.ask-msg-body');
    if (role === 'user' && text && text.includes('\n> ')) {
      // 用户消息含 markdown 引用 — 走完整 markdown 渲染让 blockquote 显示出来
      renderAssistantMarkdown(body, text);
    } else {
      body.textContent = text || '';
    }
    msgs.appendChild(wrap);
    msgs.scrollTop = msgs.scrollHeight;
    return wrap;
  }
  function appendError(msg, withSettings) {
    clearEmptyState();
    const msgs = document.getElementById('ask-ai-messages');
    const div = document.createElement('div');
    div.className = 'ask-error';
    div.innerHTML = escapeHtml(msg)
      + (withSettings ? ' <button onclick="window.AskAI && AskAI.openConfig()">检查设置</button>' : '');
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
  }
  function renderAssistantMarkdown(el, text) {
    // 较完整的 markdown: 代码块 / 标题 / 列表 / 行内 / 段落 / blockquote / hr + KaTeX 数学
    let html = escapeHtml(text || '');

    // 1. 抽出 fenced code blocks (避免内部被 markdown 二次处理)
    const codeBlocks = [];
    html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, function(_m, lang, code) {
      codeBlocks.push('<pre' + (lang ? ' class="lang-' + lang + '"' : '') + '><code>' + code + '</code></pre>');
      return ' CODEBLOCK_' + (codeBlocks.length - 1) + ' ';
    });

    // 2. 行级转换: header / hr / blockquote / list
    const lines = html.split('\n');
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      // 标题
      const hm = line.match(/^(#{1,6})\s+(.+)$/);
      if (hm) {
        const lvl = Math.min(hm[1].length, 4);  // 限制最多 h4
        out.push('<h' + lvl + '>' + hm[2] + '</h' + lvl + '>');
        i++; continue;
      }
      // 水平线
      if (/^(---|\*\*\*|___)\s*$/.test(line)) {
        out.push('<hr>');
        i++; continue;
      }
      // 无序列表 - 或 * (允许嵌套缩进, 但简化处理为扁平 ul)
      if (/^[\-\*]\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^[\-\*]\s+/.test(lines[i])) {
          items.push('<li>' + lines[i].replace(/^[\-\*]\s+/, '') + '</li>');
          i++;
        }
        out.push('<ul>' + items.join('') + '</ul>');
        continue;
      }
      // 有序列表 1. 2. 3.
      if (/^\d+\.\s+/.test(line)) {
        const items = [];
        while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
          items.push('<li>' + lines[i].replace(/^\d+\.\s+/, '') + '</li>');
          i++;
        }
        out.push('<ol>' + items.join('') + '</ol>');
        continue;
      }
      // Blockquote
      if (/^&gt;\s+/.test(line) || /^>\s+/.test(line)) {
        const quotes = [];
        while (i < lines.length && (/^&gt;\s+/.test(lines[i]) || /^>\s+/.test(lines[i]))) {
          quotes.push(lines[i].replace(/^(&gt;|>)\s+/, ''));
          i++;
        }
        out.push('<blockquote>' + quotes.join('<br>') + '</blockquote>');
        continue;
      }
      // 空行 = 段落分隔 (多个连续空行折叠成一个分隔, 不输出空字符串避免空 <p>)
      if (!line.trim()) {
        i++;
        // skip 连续多个空行
        while (i < lines.length && !lines[i].trim()) i++;
        continue;
      }
      // 普通文本行 — 累积进段落
      const para = [line];
      i++;
      while (i < lines.length && lines[i].trim()
             && !/^(#{1,6}\s|[\-\*]\s|\d+\.\s|&gt;\s|>\s)/.test(lines[i])
             && !/^(---|\*\*\*|___)\s*$/.test(lines[i])
             && !lines[i].includes(' CODEBLOCK_')) {
        para.push(lines[i]);
        i++;
      }
      out.push('<p>' + para.join('<br>') + '</p>');
    }
    html = out.join('\n');

    // 3. 行内: 粗体 / 斜体 / inline code (注意顺序: code 最先, 避免内部被加粗)
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/(?<!\w)\*([^*\n]+)\*(?!\w)/g, '<em>$1</em>');
    // 链接 [text](url)
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // 4. 还原 code blocks
    html = html.replace(/ CODEBLOCK_(\d+) /g, function(_m, idx) { return codeBlocks[+idx] || ''; });

    el.innerHTML = html;
    // 5. KaTeX 渲染数学
    if (window.renderMathInElement) {
      try {
        renderMathInElement(el, {
          delimiters: [{left: '$$', right: '$$', display: true}, {left: '\\(', right: '\\)', display: false}],
          throwOnError: false,
        });
      } catch (e) {}
    }
  }
