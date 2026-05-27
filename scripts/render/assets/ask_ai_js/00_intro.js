// =========== Ask AI widget — BYOK, OpenAI-compatible, streaming ============
// 文件按字典序拼接成单个 IIFE; 函数声明被 hoist, const/let 在 init() 被调用前完成初始化。
window.AskAI = (function() {
  const PROVIDER_DEFAULTS = {
    gemini:   { base: 'https://generativelanguage.googleapis.com/v1beta/openai', model: 'gemini-2.5-flash' },
    openai:   { base: 'https://api.openai.com/v1', model: 'gpt-4o-mini' },
    deepseek: { base: 'https://api.deepseek.com/v1', model: 'deepseek-chat' },
    ollama:   { base: 'http://localhost:11434/v1', model: 'llama3.2' },
    custom:   { base: '', model: '' },
  };
  const CONFIG_KEY = 'askAI.config';

  const S = {
    panelOpen: false,
    configOpen: false,
    selection: null,        // 实时跟踪的当前 document 选区 (随 selectionchange 变)
    pinnedSelection: null,  // 点 FAB / 打开 panel 时 snapshot 一下选区, 锁定不被后续 textarea 焦点冲掉
    messages: [],
    config: null,
    isStreaming: false,
    abortCtrl: null,
    lastSentContextHash: '',  // compact context: 跟踪上次已发送的 context, 后续轮只在变化时重发
  };
