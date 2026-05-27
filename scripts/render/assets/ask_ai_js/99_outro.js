  return {
    init: init,
    openPanel: openPanel,
    closePanel: closePanel,
    openConfig: function() { openConfig(false); },
    get state() { return S; },
  };
})();

// Init after DOM ready
if (document.readyState !== 'loading') AskAI.init();
else document.addEventListener('DOMContentLoaded', AskAI.init);
