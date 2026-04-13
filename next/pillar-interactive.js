/*!
 * pillar-interactive.js — scroll reveals, hub pillar filter, optional cost slider
 */
(function () {
  'use strict';

  if (!document.getElementById('pillar-ix-styles')) {
    var st = document.createElement('style');
    st.id = 'pillar-ix-styles';
    st.textContent =
      '@keyframes webhookLineIn{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:none}}';
    document.head.appendChild(st);
  }

  function initReveal() {
    var els = document.querySelectorAll('[data-reveal], .reveal, .reveal-left, .reveal-right');
    if (!els.length) return;
    if (!('IntersectionObserver' in window)) {
      els.forEach(function (el) {
        if (el.hasAttribute('data-reveal')) el.classList.add('is-visible');
        else el.classList.add('visible');
      });
      return;
    }
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          var t = e.target;
          if (t.hasAttribute('data-reveal')) {
            t.classList.add('is-visible');
            var d = t.getAttribute('data-reveal-delay');
            if (d) t.style.transitionDelay = d + 'ms';
          } else {
            t.classList.add('visible');
          }
        });
      },
      { threshold: 0.08, rootMargin: '0px 0px -32px 0px' }
    );
    els.forEach(function (el) {
      io.observe(el);
    });
  }

  function initPageChrome() {
    var bar = document.getElementById('progress-bar');
    var topBtn = document.getElementById('back-to-top');
    function onScroll() {
      if (bar) {
        var h = document.documentElement;
        var max = Math.max(1, h.scrollHeight - h.clientHeight);
        bar.style.width = Math.min(100, Math.max(0, (h.scrollTop / max) * 100)) + '%';
      }
      if (topBtn) topBtn.classList.toggle('visible', window.scrollY > 400);
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
    if (topBtn) {
      topBtn.addEventListener('click', function () {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }
  }

  function initStatCounters() {
    if (!('IntersectionObserver' in window)) return;
    var cntObs = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (x) {
          if (!x.isIntersecting) return;
          var el = x.target;
          var target = parseInt(el.getAttribute('data-target'), 10);
          if (isNaN(target)) return;
          var s = 0;
          var step = Math.max(1, Math.ceil(target / 55));
          var id = setInterval(function () {
            s = Math.min(s + step, target);
            el.textContent = s.toLocaleString();
            if (s >= target) clearInterval(id);
          }, 16);
          cntObs.unobserve(el);
        });
      },
      { threshold: 0.4 }
    );
    document.querySelectorAll('.stat-num[data-target]').forEach(function (el) {
      cntObs.observe(el);
    });
  }

  function initConceptChecks() {
    document.querySelectorAll('.concept-check').forEach(function (wrap) {
      var opts = wrap.querySelectorAll('.cc-opt');
      var fb = wrap.querySelector('.cc-feedback');
      opts.forEach(function (btn) {
        btn.addEventListener('click', function () {
          if (btn.disabled) return;
          var correct = btn.getAttribute('data-correct') === 'true';
          opts.forEach(function (o) {
            o.disabled = true;
            if (o.getAttribute('data-correct') === 'true') o.classList.add('is-correct');
            else if (o === btn && !correct) o.classList.add('is-wrong');
          });
          if (fb) {
            fb.classList.add('visible');
            fb.style.display = 'block';
          }
        });
      });
    });
  }

  function initHub() {
    var root = document.getElementById('pillar-hub');
    if (!root) return;

    var chips = root.querySelectorAll('.pillar-chip');
    var items = root.querySelectorAll('.deep-link-item');

    function setActive(pillar) {
      chips.forEach(function (c) {
        var on = c.getAttribute('data-pillar') === pillar;
        c.classList.toggle('is-active', on);
        c.setAttribute('aria-selected', on ? 'true' : 'false');
      });
      items.forEach(function (a) {
        var p = a.getAttribute('data-pillar');
        var show = pillar === 'all' || p === 'all' || p === pillar;
        a.style.display = show ? '' : 'none';
        a.classList.toggle('is-filtered-out', !show);
      });
    }

    chips.forEach(function (chip) {
      chip.addEventListener('click', function () {
        setActive(chip.getAttribute('data-pillar'));
      });
    });

    var diagram = root.querySelector('.pillar-diagram-svg');
    if (diagram) {
      diagram.addEventListener('mousemove', function (e) {
        var r = diagram.getBoundingClientRect();
        var x = ((e.clientX - r.left) / r.width - 0.5) * 8;
        var y = ((e.clientY - r.top) / r.height - 0.5) * 8;
        diagram.style.setProperty('--tilt-x', x + 'deg');
        diagram.style.setProperty('--tilt-y', -y + 'deg');
      });
      diagram.addEventListener('mouseleave', function () {
        diagram.style.setProperty('--tilt-x', '0deg');
        diagram.style.setProperty('--tilt-y', '0deg');
      });
    }
  }

  function initWebhookDemo() {
    var btn = document.getElementById('webhook-fire');
    var log = document.getElementById('webhook-log');
    if (!btn || !log) return;
    var processed = new Set();
    function key(p) {
      return p.tenant + ':' + p.eid;
    }
    var payload = { tenant: 't1', eid: 'evt-demo' };
    btn.addEventListener('click', function () {
      processed.clear();
      log.innerHTML = '';
      for (var i = 0; i < 3; i++) {
        (function (n) {
          setTimeout(function () {
            var kk = key(payload);
            var line = document.createElement('div');
            line.style.fontFamily = 'JetBrains Mono, monospace';
            line.style.fontSize = '12px';
            line.style.padding = '8px 0';
            line.style.borderBottom = '1px solid rgba(42,42,69,0.8)';
            line.style.animation = 'webhookLineIn 0.4s ease';
            if (processed.has(kk)) {
              line.textContent = 'Delivery ' + n + ' → duplicate_ignored';
              line.style.color = '#fbbf24';
            } else {
              processed.add(kk);
              line.textContent = 'Delivery ' + n + ' → side_effect ✓';
              line.style.color = '#34d399';
            }
            log.appendChild(line);
          }, n * 380);
        })(i + 1);
      }
    });
  }

  function initTabularToggle() {
    var root = document.getElementById('tabular-compare');
    if (!root) return;
    var btns = root.querySelectorAll('[data-show]');
    var cols = root.querySelectorAll('.compare-highlight');
    function set(mode) {
      btns.forEach(function (b) {
        b.classList.toggle('is-active', b.getAttribute('data-show') === mode);
      });
      cols.forEach(function (c) {
        c.classList.toggle('is-dim', c.getAttribute('data-col') !== mode);
      });
    }
    btns.forEach(function (b) {
      b.addEventListener('click', function () {
        set(b.getAttribute('data-show'));
      });
    });
    set('tabular');
  }

  function initAbBars() {
    var root = document.getElementById('ab-chart');
    if (!root) return;
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          root.classList.add('is-animated');
          io.disconnect();
        });
      },
      { threshold: 0.25 }
    );
    io.observe(root);
  }

  function initCostSlider(root) {
    var el = (root && root.querySelector('#cost-slider')) || document.getElementById('cost-slider');
    if (!el) return;
    var out = document.getElementById('cost-slider-output');
    var fmt = function (n) { return n.toLocaleString(undefined, { maximumFractionDigits: 0 }); };

    function calc() {
      var req = parseInt(el.value, 10);
      var price = parseFloat(el.getAttribute('data-price-per-1k')) || 0.003;
      var tin = parseInt(el.getAttribute('data-tokens-in'), 10) || 800;
      var tout = parseInt(el.getAttribute('data-tokens-out'), 10) || 400;
      var monthly = req * ((tin + tout) / 1000) * price;
      if (out) {
        out.innerHTML = '<strong>$' + fmt(monthly) + '</strong><span class="cost-sub"> / mo (illustrative)</span>';
      }
    }
    el.addEventListener('input', calc);
    calc();
  }

  function boot() {
    initPageChrome();
    initReveal();
    initStatCounters();
    initConceptChecks();
    initHub();
    initCostSlider();
    initWebhookDemo();
    initTabularToggle();
    initAbBars();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
