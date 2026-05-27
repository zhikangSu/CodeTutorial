  // ---- Prompt construction ----
  const DEFAULT_SYSTEM_PROMPT =
      '你是一个 ML 论文/源码理解助手, 帮用户阅读一份 SmolVLA 训练 trace 的可交互架构网页。\n\n'
    + '**重要: 你收到的不是 HTML 源代码** — 而是从那个页面抽出的**结构化 context**:\n'
    + '1. 用户选中的文本片段 (★ 最高优先级, 如果有)\n'
    + '2. 用户当前展开看的模块的完整详情 (qualname / shape / what / formula / 源码位置)\n'
    + '3. 页面整体拓扑 (顶层模块图 + 数据流方向 + 关键 shape) — 背景参考\n'
    + '4. 用户当前的 breadcrumb 路径 (位置 hint)\n\n'
    + '**主语判定规则 (重要)**:\n'
    + '- 如果 context 里有"选中的文本"段, **用户问题里的"这/this/它/这段/这个"默认指代那段选中文本**, 不是页面整体, 也不是当前模块。例: 选中"prefix 是模型能看到的那段 token" + 问"这是什么意思" → 回答"prefix 是什么意思", 不是"这一步是什么意思"。\n'
    + '- 如果没有选中, 主语才落到"当前展开的模块"或"breadcrumb 位置"上。\n\n'
    + '**不要在回答 / 思考里说"我没有 HTML 内容"这种话**。你有的就是上面四类结构化片段, 比 HTML 干净得多。\n'
    + 'context 没覆盖的细节: 提醒用户**点开对应模块** (drawer 展开后下一轮自动注入), 或基于通用知识直接答。\n\n'
    + '**回答风格要求**:\n'
    + '- 用 markdown: 代码块 ```...```, 行内 `code`, **粗体**, `-` / `1.` 列表\n'
    + '- 段落紧凑, **不要每句话单独成段**; 长答案才用 `###` 小标题分段, 短答案直接一两段就够\n'
    + '- 数学公式: 行内 `\\(x_t\\)` 块级 `$$...$$` (KaTeX 语法)\n'
    + '- 简洁直接、不套话; 跟当前位置无关的问题正常回答, 不强行联系当前位置';

  function buildSystemPrompt() {
    if (S.config && S.config.systemPrompt && S.config.systemPrompt.trim()) {
      return S.config.systemPrompt.trim();
    }
    return DEFAULT_SYSTEM_PROMPT;
  }

  // ---- 页面拓扑概览 (每次第一轮塞进 context, 让 AI 不会嘀咕"哪里有 HTML") ----
  function buildTopologyOverview() {
    if (typeof TREE === 'undefined' || !TREE.pipeline) return '';
    const p = TREE.pipeline;
    const lines = [];
    if (p.name) lines.push('页面: ' + p.name + (p.sub ? ' (' + p.sub + ')' : ''));
    if (p.story_line) lines.push('概念: ' + stripHtml(p.story_line));
    lines.push('');
    lines.push('拓扑 (每个 stage 一行, "/" 隔开同 stage 的并行节点):');
    const stages = p.stages || [];
    stages.forEach((stage, i) => {
      const nodeNames = (stage.nodes || []).map(id => {
        const n = TREE[id];
        if (!n) return id;
        const desc = n.sub ? ' [' + n.sub + ']' : '';
        return (n.name || id) + desc;
      }).join(' / ');
      let flag = '';
      if (stage.is_heart) flag = ' ★心脏模块';
      else if (stage.training_only) flag = ' (训练 only)';
      lines.push('  ' + (i + 1) + '. ' + nodeNames + flag);
      if (stage.outgoing_labels) {
        Object.entries(stage.outgoing_labels).forEach(([from, label]) => {
          lines.push('      ↓ ' + label);
        });
      }
    });
    return lines.join('\n');
  }
  let _topologyCache = null;
  function getTopologyOverview() {
    if (_topologyCache === null) _topologyCache = buildTopologyOverview();
    return _topologyCache;
  }

  function snapshotContext(opts) {
    opts = opts || {};
    const breadcrumb = state.path.map(p => (TREE[p] && TREE[p].name) || p).join(' ▸ ');
    const moduleId = opts.moduleId || state.drawerOpen;
    let ctx = '';

    // 注: 选中文本现在以 markdown 引用形式直接拼进 user message (见 sendMessage),
    // 不再走 context 注入 — 引用块是 LLM 最不可能忽略的显式信号。

    // === 优先级 2: 用户当前展开的模块详情 ===
    if (moduleId && TREE[moduleId]) {
      const n = TREE[moduleId];
      ctx += '## 用户当前展开看的模块\n';
      ctx += '名称: ' + (n.name || moduleId) + (n.sub ? ' · ' + n.sub : '') + '\n';
      ctx += '类型: ' + n.type + (n.functional ? ' (functional)' : '') + '\n';
      ctx += 'Qualname: ' + (n._qualname || n.qualname || '-') + '\n';
      if (n._callCount && n._callCount > 1) ctx += '调用次数: × ' + n._callCount + '\n';
      ctx += 'Shape: ' + stripHtml(n._shape_html || n.shape_html || '-') + '\n';
      if (n.what) ctx += 'What: ' + stripHtml(n.what) + '\n';
      if (n.formula) ctx += 'Formula (LaTeX): ' + n.formula + '\n';
      if (n.callout) ctx += '备注: ' + stripHtml(n.callout) + '\n';
      ctx += '源码: ' + (n._src || n.src || '-') + '\n\n';
    }

    // === 优先级 3: 整体拓扑 (背景知识, 让 AI 知道页面长什么样) ===
    const topo = getTopologyOverview();
    if (topo) ctx += '## 页面整体拓扑 (背景参考)\n' + topo + '\n\n';

    // === 优先级 4: 用户当前 breadcrumb (位置信息, 最弱的 hint) ===
    ctx += '## 用户当前位置\n';
    ctx += 'Breadcrumb: ' + breadcrumb + '\n';
    return ctx;
  }
