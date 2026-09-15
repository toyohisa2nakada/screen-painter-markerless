// Phone page: camera -> WebRTC media track to the PC, touches + clock sync over a data channel.
'use strict';
(async function () {
  const $ = (id) => document.getElementById(id);
  const video = $('video'), overlay = $('overlay'), octx = overlay.getContext('2d');
  const statusEl = $('status');
  const setStatus = (s) => { statusEl.textContent = s; };
  const nowEpoch = () => performance.timeOrigin + performance.now();

  const cfg = await fetch('/config.json').then(r => r.json());
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus('カメラAPIが使えません。HTTPSで開いていますか？');
    return;
  }

  let stream = null, peer = null, call = null, conn = null;
  let restartTimer = null;

  async function openCamera() {
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
    });
    const track = stream.getVideoTracks()[0];
    try { track.contentHint = 'detail'; } catch (_) {}
    video.srcObject = stream;
    await video.play();
    return track;
  }

  function peerOptions() {
    return {
      host: location.hostname, port: Number(location.port) || (location.protocol === 'https:' ? 443 : 80),
      path: '/peerjs', secure: location.protocol === 'https:', debug: 1,
      config: { iceServers: [] },
    };
  }

  async function tuneSender(mediaConn) {
    // raise bitrate and keep resolution (features die first when the encoder downsizes)
    for (let i = 0; i < 10; i++) {
      const pc = mediaConn.peerConnection;
      const sender = pc && pc.getSenders().find(s => s.track && s.track.kind === 'video');
      if (sender) {
        try {
          const p = sender.getParameters();
          if (!p.encodings || !p.encodings.length) p.encodings = [{}];
          p.encodings[0].maxBitrate = 2_500_000;
          p.degradationPreference = 'maintain-resolution';
          await sender.setParameters(p);
          return;
        } catch (e) { console.warn('setParameters', e); }
      }
      await new Promise(r => setTimeout(r, 300));
    }
  }

  function connectToPC() {
    if (!peer || !peer.open || !stream) return;
    if (call) { try { call.close(); } catch (_) {} }
    if (conn) { try { conn.close(); } catch (_) {} }
    setStatus('PCに接続中...');
    call = peer.call(cfg.pcPeerId, stream);
    call.on('close', () => { setStatus('映像切断 - 再接続します'); scheduleReconnect(); });
    call.on('error', (e) => { console.warn('call error', e); scheduleReconnect(); });
    tuneSender(call);
    conn = peer.connect(cfg.pcPeerId, { reliable: true, serialization: 'json' });
    conn.on('open', () => {
      setStatus('接続済み - 画面をなぞって描画');
      conn.send({ type: 'hello', ua: navigator.userAgent.slice(0, 60) });
    });
    conn.on('data', (m) => {
      if (m.type === 'ping') conn.send({ type: 'pong', t0: m.t0, tp: nowEpoch() });
    });
    conn.on('close', () => { setStatus('データ切断 - 再接続します'); scheduleReconnect(); });
    conn.on('error', (e) => console.warn('conn error', e));
  }
  function scheduleReconnect() {
    if (restartTimer) return;
    restartTimer = setTimeout(() => { restartTimer = null; connectToPC(); }, 2500);
  }

  function openPeer() {
    peer = new Peer(peerOptions());
    peer.on('open', () => connectToPC());
    peer.on('error', (e) => {
      console.warn('peer error', e.type, e);
      if (e.type === 'peer-unavailable') { setStatus('PC側ページが見つかりません - 再試行中'); scheduleReconnect(); }
      else if (e.type === 'network' || e.type === 'server-error' || e.type === 'socket-error') {
        setStatus(`シグナリングエラー (${e.type}) - 再試行中`);
        setTimeout(() => { try { peer.destroy(); } catch (_) {} openPeer(); }, 3000);
      }
    });
    peer.on('disconnected', () => setTimeout(() => { try { peer.reconnect(); } catch (_) {} }, 1000));
  }

  // ---------- touches: client coords -> video frame pixels (object-fit: contain)
  function toFrame(cx, cy) {
    const r = video.getBoundingClientRect();
    const vw = video.videoWidth || 1, vh = video.videoHeight || 1;
    const s = Math.min(r.width / vw, r.height / vh);
    const ox = r.left + (r.width - vw * s) / 2, oy = r.top + (r.height - vh * s) / 2;
    return [(cx - ox) / s, (cy - oy) / s];
  }
  const colors = ['#ff3b3b', '#3b8bff', '#3bff7a', '#ffd23b', '#ff3bd6'];
  const active = new Map(); // pointerId -> {color, last:[cx,cy]}
  function send(m) { if (conn && conn.open) conn.send(m); }
  function resizeOverlay() {
    overlay.width = overlay.clientWidth * devicePixelRatio; overlay.height = overlay.clientHeight * devicePixelRatio;
    octx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  }
  window.addEventListener('resize', resizeOverlay); resizeOverlay();
  overlay.addEventListener('pointerdown', (e) => {
    overlay.setPointerCapture(e.pointerId);
    const color = colors[active.size % colors.length];
    active.set(e.pointerId, { color, last: [e.clientX, e.clientY] });
    const [x, y] = toFrame(e.clientX, e.clientY);
    send({ type: 'touch', phase: 'start', id: e.pointerId, x, y, t: nowEpoch(), color });
  });
  overlay.addEventListener('pointermove', (e) => {
    const a = active.get(e.pointerId); if (!a) return;
    const events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    for (const ev of events) {
      const [x, y] = toFrame(ev.clientX, ev.clientY);
      send({ type: 'touch', phase: 'move', id: e.pointerId, x, y, t: nowEpoch(), color: a.color });
      octx.strokeStyle = a.color; octx.lineWidth = 3; octx.lineCap = 'round';
      octx.beginPath(); octx.moveTo(a.last[0], a.last[1]); octx.lineTo(ev.clientX, ev.clientY); octx.stroke();
      a.last = [ev.clientX, ev.clientY];
    }
  });
  const endTouch = (e) => {
    const a = active.get(e.pointerId); if (!a) return;
    const [x, y] = toFrame(e.clientX, e.clientY);
    send({ type: 'touch', phase: 'end', id: e.pointerId, x, y, t: nowEpoch(), color: a.color });
    active.delete(e.pointerId);
  };
  overlay.addEventListener('pointerup', endTouch);
  overlay.addEventListener('pointercancel', endTouch);
  $('clear').addEventListener('click', () => octx.clearRect(0, 0, overlay.width, overlay.height));

  // ---------- start (user gesture: needed for camera on some browsers)
  $('start').addEventListener('click', async () => {
    $('start').style.display = 'none';
    try {
      await openCamera();
      setStatus('カメラ起動 - シグナリング接続中');
      openPeer();
    } catch (e) {
      setStatus('カメラを開けません: ' + (e && e.message || e));
      $('start').style.display = '';
    }
  });

  // iOS stops the camera in the background: re-acquire and swap the track on return
  document.addEventListener('visibilitychange', async () => {
    if (document.visibilityState !== 'visible' || !stream) return;
    const t = stream.getVideoTracks()[0];
    if (t && t.readyState === 'live') return;
    try {
      const track = await openCamera();
      const pc = call && call.peerConnection;
      const sender = pc && pc.getSenders().find(s => s.track && s.track.kind === 'video');
      if (sender) await sender.replaceTrack(track); else connectToPC();
    } catch (e) { console.warn('re-acquire', e); }
  });
})();
