/*!
 * bg-animation.js — Neural Web v4.1
 *
 * Sweet spot between calm and interesting:
 *  · Nodes breathe slowly (size + opacity oscillation)
 *  · Depth layering — far nodes are smaller/dimmer, create parallax feel
 *  · Rare, dim, slow signal pulses (max 4, no rings/cascades on arrival)
 *  · Lines subtly brighten near cursor
 *  · Gentle cursor repulsion / attract (Shift key)
 *  · Nothing sudden or jarring
 *
 * Enhancements (v4.1):
 *  1. data-animation-density body attribute (high → more pulses, faster spawn)
 *  2. Week color theming by URL day number
 *  3. dayComplete custom event → radial particle burst
 *  4. Shift-key attract mode (flip repulsion sign)
 *  5. Constellation cluster centroids (5 slow-drifting attractors)
 */
(function () {
  'use strict';

  // ── Canvas ─────────────────────────────────────────────────────────────────
  const canvas = document.createElement('canvas');
  canvas.id = 'bg-canvas';
  canvas.style.cssText =
    'position:fixed;top:0;left:0;width:100%;height:100%;' +
    'z-index:0;pointer-events:none;opacity:0.65;';
  document.body.insertBefore(canvas, document.body.firstChild);

  const si = document.createElement('style');
  si.textContent = 'body>:not(#bg-canvas){isolation:isolate}';
  document.head.appendChild(si);

  const ctx = canvas.getContext('2d');
  let W = 0, H = 0;

  // ── Mouse ──────────────────────────────────────────────────────────────────
  const mouse = { x: -9999, y: -9999 };
  document.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });
  document.addEventListener('mouseleave', () => { mouse.x = -9999; mouse.y = -9999; });
  document.addEventListener('touchmove', e => {
    mouse.x = e.touches[0].clientX;
    mouse.y = e.touches[0].clientY;
  }, { passive: true });
  document.addEventListener('touchend', () => { mouse.x = -9999; mouse.y = -9999; });

  // === ENHANCEMENT 4: Shift-key attract mode ===
  // Holding Shift flips cursor influence from repulsion to attraction.
  let attractMode = false;
  document.addEventListener('keydown', e => { if (e.key === 'Shift') attractMode = true;  });
  document.addEventListener('keyup',   e => { if (e.key === 'Shift') attractMode = false; });

  // ── Config ─────────────────────────────────────────────────────────────────
  const CONNECT  = 125;   // max px to draw an edge
  const MOUSE_R  = 140;   // repulsion radius
  const SPEED    = 0.25;  // base drift — calm
  const DAMPING  = 0.979;
  let   MAX_PULSES = 4;   // very few at once

  // === ENHANCEMENT 1: density support ===
  // Check body attribute data-animation-density="high" for a denser animation.
  const _densityHigh = document.body.dataset.animationDensity === 'high';
  if (_densityHigh) {
    MAX_PULSES = 8;
  }
  // Spawn interval base (frames). High density → half the wait time.
  const PULSE_SPAWN_BASE = _densityHigh ? 90 : 180;
  const PULSE_SPAWN_RAND = _densityHigh ? 90 : 240;

  const COLORS = ['139,92,246', '20,184,166', '99,102,241', '167,139,250'];

  // === ENHANCEMENT 2: week color theming by URL ===
  // Parse day number from URL (e.g. "day8.html" → 8). Map to week theme.
  (function _applyWeekColors() {
    const _weekColorSets = {
      2: ['59,130,246',  '96,165,250',  '147,197,253', '37,99,235'],    // blue  #3b82f6
      3: ['245,158,11',  '251,191,36',  '217,119,6',   '253,230,138'],  // amber #f59e0b
      4: ['239,68,68',   '248,113,113', '220,38,38',   '252,165,165'],  // red   #ef4444
      5: ['13,148,136',  '20,184,166',  '15,118,110',  '94,234,212'],   // teal  #0d9488
    };
    const m = window.location.pathname.match(/day(\d+)/i) ||
              window.location.search.match(/day=(\d+)/i);
    if (!m) return;
    const day  = parseInt(m[1], 10);
    const week = day <= 7  ? 1 :
                 day <= 14 ? 2 :
                 day <= 21 ? 3 :
                 day <= 28 ? 4 : 5;
    if (week === 1 || !_weekColorSets[week]) return; // week 1 stays default
    const palette = _weekColorSets[week];
    palette.forEach((c, i) => { COLORS[i] = c; });
    COLORS.length = palette.length;
  }());

  // ── Pulse ─────────────────────────────────────────────────────────────────
  // Rare, dim, slow — just a hint that data is moving
  const pulses = [];
  let  pulseSpawnIn = PULSE_SPAWN_BASE + Math.random() * PULSE_SPAWN_RAND; // frames until next spawn

  class Pulse {
    constructor(from, to) {
      this.from  = from;
      this.to    = to;
      this.t     = 0;
      this.speed = 0.003 + Math.random() * 0.003; // 5-10 second journey
      this.color = from.color;
      this.trail = [];
    }

    update() {
      const cx = this.from.x + (this.to.x - this.from.x) * this.t;
      const cy = this.from.y + (this.to.y - this.from.y) * this.t;
      this.trail.unshift({ x: cx, y: cy });
      if (this.trail.length > 7) this.trail.pop();
      this.t += this.speed;
      return this.t >= 1;
    }

    draw() {
      const hx = this.from.x + (this.to.x - this.from.x) * Math.min(this.t, 1);
      const hy = this.from.y + (this.to.y - this.from.y) * Math.min(this.t, 1);

      // Soft fading trail
      for (let i = 1; i < this.trail.length; i++) {
        const frac = 1 - i / this.trail.length;
        ctx.beginPath();
        ctx.arc(this.trail[i].x, this.trail[i].y, 1.8 * frac, 0, 6.2832);
        ctx.fillStyle = `rgba(${this.color},${frac * 0.3})`;
        ctx.fill();
      }

      // Head — small, dim glow
      const g = ctx.createRadialGradient(hx, hy, 0, hx, hy, 7);
      g.addColorStop(0, `rgba(${this.color},0.55)`);
      g.addColorStop(1, `rgba(${this.color},0)`);
      ctx.beginPath();
      ctx.arc(hx, hy, 7, 0, 6.2832);
      ctx.fillStyle = g;
      ctx.fill();
    }
  }

  function trySpawnPulse(nodes) {
    if (pulses.length >= MAX_PULSES) return;
    // Pick a random node and one of its close neighbours
    const a = nodes[Math.floor(Math.random() * nodes.length)];
    const cd2 = CONNECT * CONNECT;
    const nb = nodes.filter(b => {
      if (b === a) return false;
      const dx = a.x - b.x, dy = a.y - b.y;
      return dx * dx + dy * dy < cd2;
    });
    if (nb.length) {
      const b = nb[Math.floor(Math.random() * nb.length)];
      // Don't duplicate an existing pulse on this edge
      const dup = pulses.some(p =>
        (p.from === a && p.to === b) || (p.from === b && p.to === a));
      if (!dup) pulses.push(new Pulse(a, b));
    }
  }

  // ── Particle ───────────────────────────────────────────────────────────────
  class Particle {
    constructor() {
      // Depth 0.4–1.0: far nodes are smaller, dimmer, slower
      this.depth      = 0.4 + Math.random() * 0.6;
      this.baseR      = (0.7 + Math.random() * 1.5) * this.depth;
      this.x          = Math.random() * W;
      this.y          = Math.random() * H;
      const spd       = SPEED * (0.5 + this.depth * 0.8);
      this.vx         = (Math.random() - 0.5) * spd * 2;
      this.vy         = (Math.random() - 0.5) * spd * 2;
      this.baseAlpha  = (0.18 + Math.random() * 0.22) * this.depth;
      this.color      = COLORS[Math.floor(Math.random() * COLORS.length)];
      // Breathing — slow sinusoidal oscillation in size & opacity
      this.breathPhase = Math.random() * Math.PI * 2;
      this.breathSpeed = 0.004 + Math.random() * 0.007; // very slow
    }

    update() {
      // Cursor repulsion (or attraction when attractMode is active — Enhancement 4)
      const dx = this.x - mouse.x;
      const dy = this.y - mouse.y;
      const d2 = dx * dx + dy * dy;
      if (d2 < MOUSE_R * MOUSE_R && d2 > 1) {
        const d  = Math.sqrt(d2);
        const f  = (1 - d / MOUSE_R) * 0.032;
        // attractMode: pull toward cursor (negate direction); default: push away
        const sign = attractMode ? -1 : 1;
        this.vx += sign * (dx / d) * f;
        this.vy += sign * (dy / d) * f;
      }

      const spd = Math.sqrt(this.vx * this.vx + this.vy * this.vy);
      const max = SPEED * 3;
      if (spd > max) { this.vx = this.vx / spd * max; this.vy = this.vy / spd * max; }

      this.vx         *= DAMPING;
      this.vy         *= DAMPING;
      this.x          += this.vx;
      this.y          += this.vy;
      this.breathPhase += this.breathSpeed;

      const m = 22;
      if (this.x < -m) this.x = W + m; else if (this.x > W + m) this.x = -m;
      if (this.y < -m) this.y = H + m; else if (this.y > H + m) this.y = -m;
    }

    draw() {
      const breath = 0.86 + 0.14 * Math.sin(this.breathPhase); // 0.86–1.0
      const dx = this.x - mouse.x, dy = this.y - mouse.y;
      const md = Math.sqrt(dx * dx + dy * dy);
      const mi = Math.max(0, 1 - md / MOUSE_R);

      const r = this.baseR * breath + mi * 1.5;
      const a = Math.min(0.6, this.baseAlpha * breath + mi * 0.15);

      ctx.beginPath();
      ctx.arc(this.x, this.y, r, 0, 6.2832);
      ctx.fillStyle = `rgba(${this.color},${a})`;
      ctx.fill();
    }
  }

  // ── Edges ──────────────────────────────────────────────────────────────────
  function drawEdges(nodes) {
    const cd2 = CONNECT * CONNECT;
    const n   = nodes.length;
    const mr  = MOUSE_R * 1.6;

    for (let i = 0; i < n - 1; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < n; j++) {
        const b  = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d2 = dx * dx + dy * dy;
        if (d2 > cd2) continue;

        const d = Math.sqrt(d2);
        // Base alpha — very faint, fades with distance
        let alpha = (1 - d / CONNECT) * 0.11;

        // Gentle boost near cursor
        const mx = (a.x + b.x) * 0.5 - mouse.x;
        const my = (a.y + b.y) * 0.5 - mouse.y;
        const md = Math.sqrt(mx * mx + my * my);
        if (md < mr) alpha += (1 - md / mr) * 0.14;

        // Slightly brighter between near-depth nodes (same layer)
        const depthDiff = Math.abs(a.depth - b.depth);
        if (depthDiff < 0.15) alpha += 0.03;

        alpha = Math.min(alpha, 0.28);

        ctx.beginPath();
        ctx.strokeStyle = `rgba(${a.color},${alpha})`;
        ctx.lineWidth   = 0.5 + (a.depth + b.depth) * 0.15; // slightly thicker for front nodes
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }
  }

  // ── Cursor aura ─────────────────────────────────────────────────────────────
  function drawAura() {
    if (mouse.x < -100) return;
    const g = ctx.createRadialGradient(mouse.x, mouse.y, 0, mouse.x, mouse.y, 100);
    g.addColorStop(0,   'rgba(139,92,246,0.06)');
    g.addColorStop(0.5, 'rgba(20,184,166,0.03)');
    g.addColorStop(1,   'rgba(0,0,0,0)');
    ctx.beginPath();
    ctx.arc(mouse.x, mouse.y, 100, 0, 6.2832);
    ctx.fillStyle = g;
    ctx.fill();
  }

  // === ENHANCEMENT 3: dayComplete event → radial particle burst ===
  // Burst particles are short-lived objects; we keep them in a separate array
  // so they never interfere with the main particle pool.
  const burstParticles = [];

  class BurstParticle {
    constructor(x, y) {
      const angle      = Math.random() * Math.PI * 2;
      const speed      = 1.5 + Math.random() * 3.5;
      this.x           = x;
      this.y           = y;
      this.vx          = Math.cos(angle) * speed;
      this.vy          = Math.sin(angle) * speed;
      this.life        = 1.0;   // 0→1, drawn while > 0
      this.decay       = 0.011 + Math.random() * 0.007; // ~1.5 s lifetime
      this.r           = 1.5 + Math.random() * 2.5;
      this.color       = COLORS[Math.floor(Math.random() * COLORS.length)];
    }

    update() {
      this.x    += this.vx;
      this.y    += this.vy;
      this.vx   *= 0.97;  // gentle drag
      this.vy   *= 0.97;
      this.life -= this.decay;
    }

    draw() {
      if (this.life <= 0) return;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r * this.life, 0, 6.2832);
      ctx.fillStyle = `rgba(${this.color},${this.life * 0.8})`;
      ctx.fill();
    }
  }

  window.addEventListener('dayComplete', () => {
    const cx = W * 0.5;
    const cy = H * 0.5;
    for (let i = 0; i < 20; i++) {
      burstParticles.push(new BurstParticle(cx, cy));
    }
  });

  // === ENHANCEMENT 5: constellation cluster centroids ===
  // Five slow-drifting centroids (one per week theme) that gently pull nearby
  // nodes toward them. The force is intentionally tiny (~0.001) so it blends
  // invisibly with the existing drift behaviour.
  const NUM_CENTROIDS = 5;
  const centroids = [];

  function initCentroids() {
    centroids.length = 0;
    for (let i = 0; i < NUM_CENTROIDS; i++) {
      // Spread starting positions evenly across canvas
      centroids.push({
        x:  (i + 0.5) * (W / NUM_CENTROIDS),
        y:  H * (0.2 + Math.random() * 0.6),
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        color: COLORS[i % COLORS.length],
      });
    }
  }

  function updateCentroids(nodes) {
    const SPRING = 0.001;          // very subtle pull
    const TOP_FRAC = 0.08;         // pull the nearest 8% of nodes

    for (const c of centroids) {
      // Drift & bounce off edges
      c.x += c.vx;
      c.y += c.vy;
      if (c.x < 0)  { c.x = 0;  c.vx = Math.abs(c.vx); }
      if (c.x > W)  { c.x = W;  c.vx = -Math.abs(c.vx); }
      if (c.y < 0)  { c.y = 0;  c.vy = Math.abs(c.vy); }
      if (c.y > H)  { c.y = H;  c.vy = -Math.abs(c.vy); }

      // Clamp drift speed to 0.1–0.2 px/frame
      const spd = Math.sqrt(c.vx * c.vx + c.vy * c.vy);
      if (spd < 0.1) { c.vx *= 0.1 / spd; c.vy *= 0.1 / spd; }
      if (spd > 0.2) { c.vx *= 0.2 / spd; c.vy *= 0.2 / spd; }

      // Find nearest N nodes and apply a soft spring toward this centroid
      const k = Math.max(1, Math.round(nodes.length * TOP_FRAC));
      // Compute distances squared, sort, take top-k
      const sorted = nodes
        .map(n => ({ n, d2: (n.x - c.x) ** 2 + (n.y - c.y) ** 2 }))
        .sort((a, b) => a.d2 - b.d2)
        .slice(0, k);

      for (const { n } of sorted) {
        n.vx += (c.x - n.x) * SPRING;
        n.vy += (c.y - n.y) * SPRING;
      }
    }
  }

  // ── Setup ──────────────────────────────────────────────────────────────────
  let particles = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function init() {
    const count = Math.max(45, Math.min(80, Math.floor(W / 16)));
    particles   = Array.from({ length: count }, () => new Particle());
    pulses.length = 0;
    initCentroids(); // Enhancement 5 — reinit centroids on resize/init
  }

  window.addEventListener('resize', () => { resize(); init(); });

  // ── Loop ───────────────────────────────────────────────────────────────────
  let paused = false;
  document.addEventListener('visibilitychange', () => { paused = document.hidden; });

  function loop() {
    requestAnimationFrame(loop);
    if (paused) return;

    ctx.clearRect(0, 0, W, H);
    drawAura();
    drawEdges(particles);

    // Enhancement 5: update centroid positions and apply spring forces
    updateCentroids(particles);

    for (const p of particles) { p.update(); p.draw(); }

    // Enhancement 3: update & draw burst particles from dayComplete event
    for (let i = burstParticles.length - 1; i >= 0; i--) {
      burstParticles[i].update();
      burstParticles[i].draw();
      if (burstParticles[i].life <= 0) burstParticles.splice(i, 1);
    }

    // Rare pulse spawning
    pulseSpawnIn--;
    if (pulseSpawnIn <= 0) {
      trySpawnPulse(particles);
      pulseSpawnIn = PULSE_SPAWN_BASE + Math.random() * PULSE_SPAWN_RAND; // every 3–7 s (or faster in high density)
    }

    // Update & draw pulses (silent arrival — no ring, no cascade)
    for (let i = pulses.length - 1; i >= 0; i--) {
      const done = pulses[i].update();
      pulses[i].draw();
      if (done) pulses.splice(i, 1);
    }
  }

  // ── Boot ───────────────────────────────────────────────────────────────────
  resize();
  init();
  loop();

}());
