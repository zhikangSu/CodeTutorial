const TREE = __TREE_JSON__;

const state = {
  path: ['pipeline'],
  drawerOpen: null,
};

function renderBreadcrumb() {
  const el = document.getElementById('breadcrumb');
  const segments = state.path.map((id, idx) => {
    const isLast = idx === state.path.length - 1;
    const name = TREE[id].name;
    if (isLast) return '<span class="current">' + name + '</span>';
    return '<a data-bc-idx="' + idx + '">' + name + '</a><span class="sep">▸</span>';
  }).join('');
  el.innerHTML = segments + '<span class="meta">batch=16 · bf16 · v18 single-GPU</span>';
  el.querySelectorAll('a[data-bc-idx]').forEach(a => {
    a.addEventListener('click', () => {
      state.path = state.path.slice(0, parseInt(a.dataset.bcIdx, 10) + 1);
      render();
    });
  });
}

function renderBlurb() {
  const el = document.getElementById('blurb');
  const cur = TREE[state.path[state.path.length - 1]];
  let html = '';
  if (state.path.length === 1 && cur.story_line) html = cur.story_line;
  else if (state.path.length > 1 && cur.blurb) html = cur.blurb;
  el.innerHTML = html;
  if (html && window.renderMathInElement) {
    renderMathInElement(el, {
      delimiters: [{left: '$$', right: '$$', display: true}, {left: '\\(', right: '\\)', display: false}]
    });
  }
}

function renderNode(id, opts) {
  opts = opts || {};
  const n = TREE[id];
  if (!n) return '<div class="node leaf">[missing: ' + id + ']</div>';
  const cls = ['node', n.type];
  if (n.functional) cls.push('functional');
  if (opts.heart || n.is_heart) cls.push('is-heart');
  if (opts.training_only) cls.push('training-only');
  const repeat = n.repeat ? '<span class="repeat-badge">×' + n.repeat + '</span>' : '';
  return '<div class="' + cls.join(' ') + '" data-id="' + id + '" tabindex="0" role="button" aria-label="' + n.name + '">'
    + '<div class="node-title">' + n.name + repeat + '</div>'
    + (n.sub ? '<div class="node-sub">' + n.sub + '</div>' : '')
    + (n.tooltip ? '<div class="tooltip">' + n.tooltip + '</div>' : '')
    + '</div>';
}

function renderRow(ids) {
  return ids.map((id, i) => (i > 0 ? '<span class="arrow-h">→</span>' : '') + renderNode(id)).join('');
}

function renderStages(stages) {
  // Vertical layout: each stage is a row; between rows draw SVG arrows + optional edge labels.
  let html = '';
  stages.forEach((stage, i) => {
    const stageClass = ['stage'];
    if (stage.training_only) stageClass.push('training-only');
    html += '<div class="' + stageClass.join(' ') + '" data-stage-idx="' + i + '">';
    stage.nodes.forEach(nodeId => {
      const heart = !!stage.is_heart;
      html += '<div class="node-slot" data-id="' + nodeId + '">' +
              renderNode(nodeId, {heart: heart, training_only: stage.training_only}) + '</div>';
    });
    html += '</div>';
    // outgoing labels container (positions resolved by JS after layout)
    if (stage.outgoing_labels) {
      html += '<div class="edge-labels" data-from-stage="' + i + '">';
      Object.entries(stage.outgoing_labels).forEach(([fromId, label]) => {
        html += '<div class="edge-label" data-from="' + fromId + '">' + label + '</div>';
      });
      html += '</div>';
    }
  });
  return '<div class="stages-wrap">'
    + '<svg class="arrows-svg" id="arrows-svg" xmlns="http://www.w3.org/2000/svg">'
    +   '<defs><marker id="arrowhead" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">'
    +     '<path d="M0,0 L8,5 L0,10 L2,5 Z" fill="#9A9080" /></marker></defs>'
    + '</svg>'
    + html
    + '</div>';
}

function renderCanvas() {
  const canvas = document.getElementById('canvas');
  const cur = TREE[state.path[state.path.length - 1]];

  if (cur.layout === 'stages' && cur.stages) {
    canvas.innerHTML = renderStages(cur.stages);
    canvas.querySelectorAll('.node').forEach(el => attachNodeHandlers(el));
    // Compute arrow positions after browser layout
    requestAnimationFrame(() => { drawStageArrows(cur.stages); });
    // Redraw on window resize
    if (!window._smolvlaArrowsBound) {
      window._smolvlaArrowsBound = true;
      window.addEventListener('resize', () => {
        const c = TREE[state.path[state.path.length - 1]];
        if (c.layout === 'stages' && c.stages) drawStageArrows(c.stages);
      });
    }
  } else {
    const children = cur.children || [];
    canvas.innerHTML = '<div class="row">' + renderRow(children) + '</div>';
    canvas.querySelectorAll('.node').forEach(el => attachNodeHandlers(el));
  }
}

function attachNodeHandlers(el) {
  el.addEventListener('click', () => handleNodeClick(el.dataset.id));
  el.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleNodeClick(el.dataset.id); }
  });
}

function drawStageArrows(stages) {
  const svg = document.getElementById('arrows-svg');
  if (!svg) return;
  const wrap = svg.parentElement;
  const wrapRect = wrap.getBoundingClientRect();
  // Size svg to overlay full wrap
  svg.setAttribute('width', wrapRect.width);
  svg.setAttribute('height', wrapRect.height);
  svg.setAttribute('viewBox', '0 0 ' + wrapRect.width + ' ' + wrapRect.height);
  // Clear old paths but keep defs
  Array.from(svg.querySelectorAll('path.arrow-line, text.shape-label')).forEach(e => e.remove());

  const stageEls = wrap.querySelectorAll('.stage');
  for (let i = 0; i < stageEls.length - 1; i++) {
    const fromNodes = Array.from(stageEls[i].querySelectorAll('.node'));
    const toNodes = Array.from(stageEls[i + 1].querySelectorAll('.node'));
    const fromTrainingOnly = stageEls[i].classList.contains('training-only');
    const toTrainingOnly = stageEls[i + 1].classList.contains('training-only');
    const dashed = fromTrainingOnly || toTrainingOnly;
    fromNodes.forEach(fn => {
      const fr = fn.getBoundingClientRect();
      const fx = fr.left + fr.width / 2 - wrapRect.left;
      const fy = fr.bottom - wrapRect.top;
      toNodes.forEach(tn => {
        const tr = tn.getBoundingClientRect();
        const tx = tr.left + tr.width / 2 - wrapRect.left;
        const ty = tr.top - wrapRect.top;
        const midY = (fy + ty) / 2;
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', 'M ' + fx + ',' + fy + ' C ' + fx + ',' + midY + ' ' + tx + ',' + midY + ' ' + tx + ',' + ty);
        path.setAttribute('stroke', '#9A9080');
        path.setAttribute('stroke-width', '1.4');
        path.setAttribute('fill', 'none');
        path.setAttribute('class', 'arrow-line');
        path.setAttribute('marker-end', 'url(#arrowhead)');
        if (dashed) path.setAttribute('stroke-dasharray', '5,4');
        svg.appendChild(path);
      });
    });
  }

  // Position edge labels (HTML divs) near their from-node bottom
  const labelGroups = wrap.querySelectorAll('.edge-labels');
  labelGroups.forEach(g => {
    Array.from(g.querySelectorAll('.edge-label')).forEach(lb => {
      const fromId = lb.dataset.from;
      const fromNode = wrap.querySelector('.node[data-id="' + fromId + '"]');
      if (!fromNode) return;
      const fr = fromNode.getBoundingClientRect();
      const cx = fr.left + fr.width / 2 - wrapRect.left;
      const by = fr.bottom - wrapRect.top;
      lb.style.left = cx + 'px';
      lb.style.top = (by + 8) + 'px';
    });
  });
}

function handleNodeClick(id) {
  const n = TREE[id];
  if (!n) return;
  if (n.type === 'composite') {
    state.path.push(id);
    render();
  } else {
    openDrawer(id);
  }
}

function openDrawer(id) {
  const n = TREE[id];
  const callCountBadge = n._callCount && n._callCount > 1
    ? '<span class="call-count">× ' + n._callCount + '</span>' : '';
  const functionalBadge = n.functional ? '<span class="badge functional">functional</span>' : '';
  const repeatBadge = n.repeat ? '<span class="badge">part of × ' + n.repeat + ' repeated block</span>' : '';
  // trained/frozen badge — based on whether this leaf received backward grads in the trace
  let trainBadge = '';
  if (n._trained === true) {
    const bwdN = n._backward_count || 1;
    trainBadge = '<span class="badge trained" title="该模块在 trace 里收到 ' + bwdN + ' 次梯度">✓ trained · ' + bwdN + ' grad' + (bwdN > 1 ? 's' : '') + '</span>';
  } else if (n._trained === false) {
    trainBadge = '<span class="badge frozen" title="该模块在 trace 里没收到任何梯度, 推测 requires_grad=False">✗ frozen</span>';
  }

  const content = '<h2>' + n.name + (n.sub ? ' <span style="font-family: JetBrains Mono, monospace; font-size: 13px; color: var(--text-3); font-weight: 400;">· ' + n.sub + '</span>' : '') + '</h2>'
    + '<div class="qualname">' + (n._qualname || n.qualname || '') + callCountBadge + '</div>'
    + ((functionalBadge || repeatBadge || trainBadge) ? '<div class="badge-row">' + functionalBadge + repeatBadge + trainBadge + '</div>' : '')
    + '<h3>Shape</h3>'
    + '<div class="shape-box">' + (n._shape_html || n.shape_html || '—') + '</div>'
    + '<h3>What it does</h3>'
    + '<p>' + (n.what || '') + '</p>'
    + (n._act_chart_svg ? n._act_chart_svg : '')
    + (n.formula
        ? '<h3>Formula</h3><div class="formula" data-katex-pre="' + (n._formulaHtml ? '1' : '0') + '">'
          + (n._formulaHtml ? n._formulaHtml : '$$' + n.formula + '$$')
          + '</div>'
        : '')
    + (n.callout ? '<div class="callout">' + n.callout + '</div>' : '')
    + '<h3>Source</h3>'
    + '<div class="src-link">' + (n._src_link
        ? '<a href="' + n._src_link + '" title="open in VSCode">' + (n._src || n.src || '—') + '</a>'
        : (n._src || n.src || '—')) + '</div>'
    + (n._trace_src && n._trace_src !== n._src
        ? '<div class="src-link src-link-trace">forward 实际执行处: '
          + (n._trace_src_link ? '<a href="' + n._trace_src_link + '">' + n._trace_src + '</a>' : n._trace_src)
          + '</div>'
        : '')
    + '<div class="drawer-actions">'
    +   '<button class="drawer-action-btn" onclick="copyModuleContext(\'' + id + '\')" title="把模块完整上下文复制成 markdown, 粘贴到 ChatGPT / Claude 网页用">'
    +     '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="4" y="3" width="9" height="11" rx="1"/><path d="M3 12V2h9"/></svg>'
    +     ' <span>复制上下文</span>'
    +   '</button>'
    +   '<button class="drawer-action-btn primary" onclick="window.AskAI && AskAI.openPanel({moduleId: \'' + id + '\'})" title="基于此模块向 AI 提问 (Ctrl+I 直接打开问答面板)">'
    +     '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M3 12 L13 4 M13 4 L9 4 M13 4 L13 8"/><circle cx="4" cy="13" r="1.2" fill="currentColor"/></svg>'
    +     ' <span>问 AI</span>'
    +   '</button>'
    + '</div>';
  document.getElementById('drawer-content').innerHTML = content;
  document.getElementById('drawer').classList.add('open');
  document.getElementById('drawer').setAttribute('aria-hidden', 'false');
  document.getElementById('backdrop').classList.add('open');
  state.drawerOpen = id;
  if (window.renderMathInElement) {
    // formula 已预渲的话 data-katex-pre="1", 不重复处理; 其余 (inline \( \) 在 what/callout) 仍走 auto-render
    renderMathInElement(document.getElementById('drawer-content'), {
      delimiters: [{left: '$$', right: '$$', display: true}, {left: '\\(', right: '\\)', display: false}],
      ignoredClasses: ['katex-pre-rendered'],
    });
  }
}

function closeDrawer() {
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('drawer').setAttribute('aria-hidden', 'true');
  document.getElementById('backdrop').classList.remove('open');
  state.drawerOpen = null;
}

function render() {
  renderBreadcrumb();
  renderBlurb();
  renderCanvas();
}

// ---- Helpers for "Copy context" + Ask AI integration ----
function stripHtml(html) {
  if (!html) return '';
  const tmp = document.createElement('div');
  tmp.innerHTML = String(html).replace(/<br\s*\/?>/gi, '\n').replace(/&nbsp;/g, ' ');
  return (tmp.textContent || tmp.innerText || '').replace(/\s+\n/g, '\n').trim();
}

function buildModuleContextMarkdown(id) {
  const n = TREE[id];
  if (!n) return '';
  const breadcrumb = state.path.map(p => (TREE[p] && TREE[p].name) || p).join(' ▸ ');
  const lines = [];
  lines.push('# ' + (n.name || id) + (n.sub ? ' · ' + n.sub : ''));
  lines.push('');
  lines.push('**位置**: ' + breadcrumb);
  if (n._qualname || n.qualname) lines.push('**Qualname**: `' + (n._qualname || n.qualname) + '`');
  if (n._callCount && n._callCount > 1) lines.push('**调用次数**: × ' + n._callCount);
  if (n.functional) lines.push('**类型**: functional (非 nn.Module)');
  lines.push('**Shape**: ' + stripHtml(n._shape_html || n.shape_html || '—'));
  if (n.what) { lines.push(''); lines.push('## 说明'); lines.push(stripHtml(n.what)); }
  if (n.formula) { lines.push(''); lines.push('## 公式'); lines.push('$$' + n.formula + '$$'); }
  if (n.callout) { lines.push(''); lines.push('## 备注'); lines.push(stripHtml(n.callout)); }
  lines.push('');
  lines.push('**源码**: ' + (n._src || n.src || '—'));
  if (n._trace_src && n._trace_src !== n._src) {
    lines.push('**forward 实际执行处**: ' + n._trace_src);
  }
  return lines.join('\n');
}

function copyModuleContext(id) {
  const md = buildModuleContextMarkdown(id);
  navigator.clipboard.writeText(md).then(
    () => flashToast('已复制此模块的完整上下文 (可粘贴到 ChatGPT / Claude 网页)'),
    () => flashToast('复制失败 — 浏览器拒绝访问剪贴板', true)
  );
}

function flashToast(msg, isError) {
  let t = document.getElementById('toast');
  if (!t) {
    t = document.createElement('div');
    t.id = 'toast';
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.className = 'toast' + (isError ? ' error' : '') + ' show';
  clearTimeout(t._timer);
  t._timer = setTimeout(() => { t.className = 'toast'; }, 2400);
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    // Ask AI panel 优先吃 Esc, 然后是 module drawer, 最后是 breadcrumb
    if (window.AskAI && AskAI.state.panelOpen) { AskAI.closePanel(); return; }
    if (state.drawerOpen) { closeDrawer(); return; }
    if (state.path.length > 1) { state.path.pop(); render(); }
  }
  // Ctrl/Cmd + I: 直接打开 Ask AI 面板
  if ((e.ctrlKey || e.metaKey) && (e.key === 'i' || e.key === 'I')) {
    e.preventDefault();
    if (window.AskAI) AskAI.openPanel({});
  }
});
document.getElementById('drawer-close').addEventListener('click', closeDrawer);
document.getElementById('backdrop').addEventListener('click', closeDrawer);

// KaTeX 预热 + 预渲所有 leaf formula 到 _formulaHtml 缓存。
// 跑在 idle 时段, 避免开第一个抽屉时撞 50-100ms 的 JIT + macro 编译延迟。
function preRenderKatex() {
  if (!window.katex) return;
  // 1) 暖一次, 让 KaTeX 把所有内部 lookup table 实例化
  try { window.katex.renderToString('x + y', {throwOnError: false}); } catch (e) {}
  // 2) 把每个 leaf 的 formula 字段提前渲成 HTML 字符串, 缓存进 TREE
  for (const id in TREE) {
    const n = TREE[id];
    if (n && n.formula && !n._formulaHtml) {
      try {
        n._formulaHtml = window.katex.renderToString(n.formula, {
          displayMode: true, throwOnError: false,
        });
      } catch (e) { /* leave undefined; openDrawer falls back to live render */ }
    }
  }
}
// KaTeX 是 defer 加载, 不保证 render() 时已 ready — 用 requestIdleCallback 兜底, 没有就 setTimeout
if (window.requestIdleCallback) {
  requestIdleCallback(preRenderKatex, {timeout: 1500});
} else {
  setTimeout(preRenderKatex, 200);
}

render();
