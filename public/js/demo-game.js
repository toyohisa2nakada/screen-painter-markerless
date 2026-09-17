// A small animated "game" scene drawn on a canvas. It is the thing the phone films.
// The texture layer is optional: flat UI is the worst case for feature matching, so the
// PC page lets you toggle it to see the difference.
(function (root) {
  'use strict';

  class DemoGame {
    constructor(canvas, opts = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.W = canvas.width; this.H = canvas.height;
      this.texture = opts.texture !== false;
      this.frozen = !!opts.frozen; // for tests: no motion
      this.t0 = performance.now();
      this.balls = [];
      const rnd = mulberry32(12345);
      for (let i = 0; i < 12; i++) {
        this.balls.push({
          x: rnd() * this.W, y: rnd() * this.H, r: 18 + rnd() * 40,
          vx: (rnd() - 0.5) * 60, vy: (rnd() - 0.5) * 60,
          hue: (i * 37) % 360, label: String.fromCharCode(65 + i),
        });
      }
      this.texCanvas = document.createElement('canvas');
      this.texCanvas.width = this.W; this.texCanvas.height = this.H;
      this._buildTexture(rnd);
      this.last = performance.now();
    }

    _buildTexture(rnd) {
      // low-contrast random shapes: cheap to draw once, rich in corners.
      const c = this.texCanvas.getContext('2d');
      c.fillStyle = '#1b2230'; c.fillRect(0, 0, this.W, this.H);
      for (let i = 0; i < 900; i++) {
        const x = rnd() * this.W, y = rnd() * this.H, s = 4 + rnd() * 26;
        c.fillStyle = `hsl(${(rnd() * 360) | 0} 30% ${18 + rnd() * 26}%)`;
        c.beginPath();
        if (rnd() < 0.5) c.rect(x, y, s, s * (0.5 + rnd()));
        else { c.moveTo(x, y); c.lineTo(x + s, y + s * 0.3); c.lineTo(x + s * 0.2, y + s); }
        c.fill();
      }
      c.strokeStyle = 'rgba(255,255,255,0.10)'; c.lineWidth = 1.5;
      for (let i = 0; i < 120; i++) {
        c.beginPath(); c.moveTo(rnd() * this.W, rnd() * this.H); c.lineTo(rnd() * this.W, rnd() * this.H); c.stroke();
      }
    }

    step(dt) {
      for (const b of this.balls) {
        b.x += b.vx * dt; b.y += b.vy * dt;
        if (b.x < b.r || b.x > this.W - b.r) { b.vx *= -1; b.x = Math.min(Math.max(b.x, b.r), this.W - b.r); }
        if (b.y < b.r || b.y > this.H - b.r) { b.vy *= -1; b.y = Math.min(Math.max(b.y, b.r), this.H - b.r); }
      }
    }

    draw(now) {
      const ctx = this.ctx;
      const dt = Math.min(0.05, (now - this.last) / 1000); this.last = now;
      if (!this.frozen) this.step(dt);
      if (this.texture) ctx.drawImage(this.texCanvas, 0, 0);
      else { ctx.fillStyle = '#1b2230'; ctx.fillRect(0, 0, this.W, this.H); }
      for (const b of this.balls) {
        ctx.beginPath(); ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
        ctx.fillStyle = `hsl(${b.hue} 70% 55%)`; ctx.fill();
        ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.stroke();
        ctx.fillStyle = '#111'; ctx.font = `bold ${b.r}px sans-serif`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(b.label, b.x, b.y);
      }
      ctx.fillStyle = '#fff'; ctx.font = '28px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText('Screen Painter - markerless demo', 24, 20);
      ctx.font = '20px monospace';
      ctx.fillText(((now - this.t0) / 1000).toFixed(1) + ' s', 24, 58);
    }
  }

  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  root.DemoGame = DemoGame;
})(window);
