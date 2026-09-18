// PC page: renders the demo game, receives the phone camera over WebRTC (PeerJS),
// estimates where the screen is in the phone image (XFeat + RANSAC homography) and paints
// the phone's touches onto the screen.
'use strict';
(async function () {
  const $ = (id) => document.getElementById(id);
  const gameCanvas = $('game'), paintCanvas = $('paint'), hudCanvas = $('hud');
  const GW = gameCanvas.width, GH = gameCanvas.height;
  const paint = paintCanvas.getContext('2d');
  const hud = hudCanvas.getContext('2d');
  const debugCanvas = $('debug'), dbg = debugCanvas.getContext('2d');
  const scratch = document.createElement('canvas');   // reference grabs
  const scratch2 = document.createElement('canvas');  // phone grabs
  const statusEl = $('status'), statsEl = $('stats');
  const setStatus = (s) => { statusEl.textContent = s; };

  // ---------- settings
  const S = {
    refLevels: [1280, 640, 320, 160],   // reference pyramid (max side in px). XFeat is single-scale.
    refTopK: 1500, phoneTopK: 1024,
    phoneMaxSide: 640,
    refIntervalMs: 120, refKeep: 16,
    ransacThr: 3,                  // px in the phone frame
    minCos: 0.82,
    get minInliers() { return Number($('minInl').value); },
    get latencyMs() { return Number($('latency').value); },
    get smooth() { return Number($('smooth').value) / 100; },
    get showDebug() { return $('showDebug').checked; },
  };
  for (const [id, out] of [['latency', 'latencyV'], ['minInl', 'minInlV'], ['smooth', 'smoothV']]) {
    $(id).addEventListener('input', () => { $(out).textContent = $(id).value; });
  }
  $('toggle').addEventListener('click', () => $('panel').classList.toggle('hidden'));
  window.addEventListener('keydown', (e) => { if (e.key === 'd' || e.key === 'D') $('panel').classList.toggle('hidden'); });
  $('clear').addEventListener('click', () => paint.clearRect(0, 0, GW, GH));

  // ---------- demo game
  const urlp = new URLSearchParams(location.search);
  const game = new DemoGame(gameCanvas, { texture: true, frozen: urlp.get('frozen') === '1' });
  $('texture').addEventListener('change', () => { game.texture = $('texture').checked; });
  function gameLoop(now) { game.draw(now); requestAnimationFrame(gameLoop); }
  requestAnimationFrame(gameLoop);

  // ---------- config, QR
  const cfg = await fetch('/config.json').then(r => r.json());
  let host = location.hostname;
  if ((host === 'localhost' || host === '127.0.0.1') && cfg.lanAddresses.length) host = cfg.lanAddresses[0];
  const phoneUrl = `${location.protocol}//${host}:${location.port || (location.protocol === 'https:' ? 443 : 80)}/phone.html`;
  $('phoneUrl').textContent = phoneUrl;
  $('qr').src = '/qr.svg?text=' + encodeURIComponent(phoneUrl);
  if (location.protocol !== 'https:') setStatus('注意: HTTPSではありません。スマホのカメラはHTTPSが必要です (README: mkcert)。');

  // ---------- feature extractor
  const client = new Matcher.FeatureClient();
  setStatus('モデル読み込み中...');
  await client.init('/models/xfeat_web.onnx', urlp.get('gpu') !== '0');
  setStatus(`モデル準備完了 (backend: ${client.backend})。スマホの接続待ち...`);

  // ---------- reference (screen) features: ring buffer of {t0, t1, feat}
  const refs = [];
  let refBusy = false, lastRefGray = null;
  async function updateReference() {
    if (refBusy) return;
    refBusy = true;
    const t = performance.now();
    try {
      // cheap change detection on the smallest level
      const small = Matcher.fitSize(GW, GH, 320);
      const sg = Matcher.grabGray(gameCanvas, GW, GH, small.W, small.H, scratch);
      let changed = true;
      if (lastRefGray && lastRefGray.length === sg.gray.length) {
        let d = 0; const g = sg.gray, l = lastRefGray;
        for (let i = 0; i < g.length; i += 7) d += Math.abs(g[i] - l[i]);
        changed = d / (g.length / 7) > 1.0;
      }
      if (!changed && refs.length) { refs[refs.length - 1].t1 = t; return; }
      lastRefGray = sg.gray.slice();
      const sets = [];
      let msNet = 0;
      for (const L of S.refLevels) {
        const sz = Matcher.fitSize(GW, GH, L);
        const g = (L === 320) ? { gray: sg.gray, sx: sg.sx, sy: sg.sy } : Matcher.grabGray(gameCanvas, GW, GH, sz.W, sz.H, scratch);
        const f = await client.extract(g.gray, sz.W, sz.H, { topK: S.refTopK, tag: 'ref' });
        msNet += f.msNet + f.msPost;
        sets.push(Matcher.toFrame(f, g.sx, g.sy));
      }
      refs.push({ t0: t, t1: performance.now(), feat: Matcher.concat(sets), ms: msNet });
      while (refs.length > S.refKeep) refs.shift();
    } catch (e) { console.error('reference', e); }
    finally { refBusy = false; }
  }
  setInterval(updateReference, S.refIntervalMs);

  function pickRefs(tc, max = 3) {
    if (!refs.length) return [];
    const scored = refs.map(r => ({ r, d: tc < r.t0 ? r.t0 - tc : tc > r.t1 ? tc - r.t1 : 0 }));
    scored.sort((a, b) => a.d - b.d);
    return scored.slice(0, max).map(s => s.r);
  }

  // ---------- clock sync + data channel
  let dataConn = null;
  let clock = { offset: 0, rtt: 50, samples: [] }; // phoneTime = pcTime + offset (both in epoch ms)
  const nowEpoch = () => performance.timeOrigin + performance.now();
  const epochToPerf = (e) => e - performance.timeOrigin;
  function onData(msg) {
    if (msg.type === 'pong') {
      const t2 = nowEpoch(), rtt = t2 - msg.t0;
      clock.samples.push({ rtt, offset: msg.tp - (msg.t0 + t2) / 2 });
      if (clock.samples.length > 12) clock.samples.shift();
      const best = clock.samples.reduce((a, b) => (b.rtt < a.rtt ? b : a));
      clock.rtt = best.rtt; clock.offset = best.offset;
    } else if (msg.type === 'touch') {
      onTouch(msg);
    } else if (msg.type === 'hello') {
      setStatus(`スマホ接続: ${msg.ua || ''}`);
    }
  }
  setInterval(() => { if (dataConn && dataConn.open) dataConn.send({ type: 'ping', t0: nowEpoch() }); }, 700);

  // ---------- homography state
  // Hcur: phone frame px -> game px (smoothed). quad: phone corners in game px.
  let Hcur = null, quadCur = null, centerCur = null, lastGoodAt = 0;
  const Hhist = []; // {t (perf ms, capture time), H}
  let lastEst = null;

  function acceptH(H, tc, vw, vh, nInl) {
    const q = Matcher.quadOf(H, vw, vh);
    if (!Matcher.quadIsSane(q, 1e5)) return false;
    // smoothing on the projected phone corners, reset on big jumps
    const a = S.smooth;
    if (quadCur && performance.now() - lastGoodAt < 500) {
      let jump = 0;
      for (let i = 0; i < 4; i++) jump = Math.max(jump, Math.hypot(q[i][0] - quadCur[i][0], q[i][1] - quadCur[i][1]));
      if (jump < 0.35 * GW) {
        for (let i = 0; i < 4; i++) {
          q[i][0] = a * quadCur[i][0] + (1 - a) * q[i][0];
          q[i][1] = a * quadCur[i][1] + (1 - a) * q[i][1];
        }
      }
    }
    quadCur = q;
    Hcur = Homography.fromPoints(new Float32Array([0, 0, vw, 0, vw, vh, 0, vh]),
      new Float32Array([q[0][0], q[0][1], q[1][0], q[1][1], q[2][0], q[2][1], q[3][0], q[3][1]]));
    centerCur = Homography.apply(Hcur, vw / 2, vh / 2);
    lastGoodAt = performance.now();
    Hhist.push({ t: tc, H: Hcur });
    while (Hhist.length > 20) Hhist.shift();
    return true;
  }

  function homographyAt(tPerf) {
    if (!Hhist.length) return null;
    let best = Hhist[Hhist.length - 1];
    for (const h of Hhist) if (Math.abs(h.t - tPerf) < Math.abs(best.t - tPerf)) best = h;
    return best.H;
  }

  // ---------- touches -> paint
  const strokes = new Map(); // touch id -> last point in game px
  function onTouch(m) {
    const tPerf = epochToPerf(m.t - clock.offset);
    const H = homographyAt(tPerf) || Hcur;
    if (!H) return;
    const [gx, gy] = Homography.apply(H, m.x, m.y);
    if (m.phase === 'start') { strokes.set(m.id, [gx, gy]); return; }
    const prev = strokes.get(m.id);
    if (prev) {
      paint.strokeStyle = m.color || '#ff3b3b'; paint.lineWidth = 6; paint.lineCap = 'round'; paint.lineJoin = 'round';
      paint.beginPath(); paint.moveTo(prev[0], prev[1]); paint.lineTo(gx, gy); paint.stroke();
    }
    if (m.phase === 'end') strokes.delete(m.id); else strokes.set(m.id, [gx, gy]);
  }

  // ---------- phone video processing
  const video = document.createElement('video');
  video.muted = true; video.playsInline = true; video.autoplay = true;
  let frameBusy = false;
  const stats = { fps: 0, frames: 0, lastFpsAt: performance.now(), msExtract: 0, msMatch: 0, inliers: 0, matches: 0, refTried: 0 };

  async function processFrame(now, meta) {
    if (frameBusy || !refs.length) return;
    frameBusy = true;
    // console.log(now - meta.captureTime)
    try {
      const vw = video.videoWidth, vh = video.videoHeight;
      if (!vw || !vh) return;
      // estimated capture time on the PC clock (perf ms)
      let tc = now - clock.rtt / 2 - S.latencyMs;
      if (meta && meta.captureTime) tc = meta.captureTime; // Chrome fills this for remote streams when available
      const sz = Matcher.fitSize(vw, vh, S.phoneMaxSide);
      const g = Matcher.grabGray(video, vw, vh, sz.W, sz.H, scratch2);
      const t0 = performance.now();
      const pfRaw = await client.extract(g.gray, sz.W, sz.H, { topK: S.phoneTopK, tag: 'phone' });
      const pf = Matcher.toFrame(pfRaw, g.sx, g.sy);
      const t1 = performance.now();
      let best = null, tried = 0;
      for (const r of pickRefs(tc, 3)) {
        tried++;
        const est = await Matcher.estimate(pf, r.feat, { client, threshold: S.ransacThr, minCos: S.minCos });
        if (!best || est.nInliers > best.nInliers) best = est;
        if (est.nInliers >= Math.max(S.minInliers, 40)) break;
      }
      const t2 = performance.now();
      let ok = false;
      if (best && best.H && best.nInliers >= S.minInliers) ok = acceptH(best.H, tc, vw, vh, best.nInliers);
      lastEst = { est: best, ok, vw, vh, pf };
      stats.msExtract = t1 - t0; stats.msMatch = t2 - t1; stats.refTried = tried;
      stats.inliers = best ? best.nInliers : 0; stats.matches = best ? best.nMatches : 0;
      stats.frames++;
      if (now - stats.lastFpsAt > 1000) { stats.fps = stats.frames * 1000 / (now - stats.lastFpsAt); stats.frames = 0; stats.lastFpsAt = now; }
      drawDebug();
    } catch (e) { console.error('frame', e); }
    finally { frameBusy = false; }
  }
  function frameLoop(now, meta) {
    processFrame(now, meta);
    video.requestVideoFrameCallback(frameLoop);
  }

  // ---------- debug drawing
  function drawDebug() {
    const tracking = Hcur && performance.now() - lastGoodAt < 700;
    hud.clearRect(0, 0, GW, GH);
    if (S.showDebug && tracking && quadCur) {
      hud.strokeStyle = 'rgba(0,255,180,0.9)'; hud.lineWidth = 4; hud.setLineDash([12, 8]);
      hud.beginPath(); hud.moveTo(quadCur[0][0], quadCur[0][1]);
      for (let i = 1; i < 4; i++) hud.lineTo(quadCur[i][0], quadCur[i][1]);
      hud.closePath(); hud.stroke(); hud.setLineDash([]);

      const [cx, cy] = centerCur;
      hud.strokeStyle = 'rgba(60,255,60,0.95)'; hud.lineWidth = 3;
      hud.beginPath();
      hud.moveTo(cx - 20, cy); hud.lineTo(cx + 20, cy);
      hud.moveTo(cx, cy - 20); hud.lineTo(cx, cy + 20);
      hud.stroke();
    }
    statsEl.textContent =
      `${tracking ? 'TRACKING' : 'lost'}  fps ${stats.fps.toFixed(1)}  backend ${client.backend}\n` +
      `extract ${stats.msExtract.toFixed(0)} ms  match+ransac ${stats.msMatch.toFixed(0)} ms (refs tried ${stats.refTried})\n` +
      `matches ${stats.matches}  inliers ${stats.inliers}  refs ${refs.length} (${refs.length ? refs[refs.length - 1].ms.toFixed(0) : 0} ms/ref)\n` +
      `rtt ${clock.rtt.toFixed(0)} ms  clock offset ${clock.offset.toFixed(0)} ms`;
    if (!S.showDebug || !lastEst) return;
    const { est, vw, vh, pf } = lastEst;
    const dw = debugCanvas.width, dh = Math.round(dw * vh / vw);
    if (debugCanvas.height !== dh) debugCanvas.height = dh;
    dbg.drawImage(video, 0, 0, dw, dh);
    const s = dw / vw;
    dbg.fillStyle = 'rgba(255,255,0,0.5)';
    for (let i = 0; i < pf.count; i++) dbg.fillRect(pf.kpts[i * 2] * s - 1, pf.kpts[i * 2 + 1] * s - 1, 2, 2);
    if (est && est.H) {
      dbg.fillStyle = 'rgba(0,255,0,0.9)';
      for (let k = 0; k < est.nInliers; k++) {
        const i = est.inliers[k];
        dbg.fillRect(est.src[i * 2] * s - 1.5, est.src[i * 2 + 1] * s - 1.5, 3, 3);
      }
    }
    if (tracking) {
      const Hinv = Homography.invert(Hcur);
      if (Hinv) {
        dbg.strokeStyle = 'rgba(0,255,180,0.9)'; dbg.lineWidth = 2;
        dbg.beginPath();
        for (const [i, [x, y]] of [[0, 0], [GW, 0], [GW, GH], [0, GH]].entries()) {
          const p = Homography.apply(Hinv, x, y);
          if (i === 0) dbg.moveTo(p[0] * s, p[1] * s); else dbg.lineTo(p[0] * s, p[1] * s);
        }
        dbg.closePath(); dbg.stroke();
      }
    }
  }

  // ---------- PeerJS
  const peerOpts = {
    host: location.hostname, port: Number(location.port) || (location.protocol === 'https:' ? 443 : 80),
    path: '/peerjs', secure: location.protocol === 'https:', debug: 1,
    config: { iceServers: [] }, // same LAN: host candidates only
  };
  function openPeer() {
    const peer = new Peer(cfg.pcPeerId, peerOpts);
    peer.on('open', (id) => setStatus(`Peer "${id}" 待受中。スマホでQRを開いてください。`));
    peer.on('error', (e) => {
      console.warn('peer error', e.type, e);
      if (e.type === 'unavailable-id' || e.type === 'network' || e.type === 'server-error') {
        setStatus(`Peer error: ${e.type} - 3秒後に再試行`);
        setTimeout(() => { try { peer.destroy(); } catch (_) { } openPeer(); }, 3000);
      }
    });
    peer.on('disconnected', () => { setStatus('シグナリング切断 - 再接続中'); setTimeout(() => peer.reconnect(), 1000); });
    peer.on('connection', (conn) => {
      dataConn = conn;
      conn.on('data', onData);
      conn.on('close', () => { if (dataConn === conn) dataConn = null; setStatus('スマホ切断'); });
    });
    peer.on('call', (call) => {
      call.answer(); // receive only
      call.on('stream', (stream) => {
        video.srcObject = stream;
        video.play().then(() => {
          setStatus('映像受信中');
          video.requestVideoFrameCallback(frameLoop);
        });
      });
      call.on('close', () => { setStatus('映像切断'); video.srcObject = null; });
    });
  }
  openPeer();
  window.__sp = { refs, stats, get Hcur() { return Hcur; }, get quad() { return quadCur; }, paintCanvas, clock, get lastEst() { return lastEst; } }; // test hook
})();
