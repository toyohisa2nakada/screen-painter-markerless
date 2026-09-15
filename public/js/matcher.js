// Main-thread helpers: a promise-based client for feature-worker.js, image prep and
// feature-set -> homography estimation.
(function (root) {
  'use strict';

  class FeatureClient {
    constructor() {
      this.worker = new Worker('/js/feature-worker.js');
      this.pending = new Map();
      this.nextId = 1;
      this.backend = null;
      this.worker.onmessage = (ev) => {
        const m = ev.data;
        if (m.type === 'ready') { this.backend = m.backend; this._ready && this._ready(m); return; }
        const p = this.pending.get(m.id);
        if (!p) return;
        this.pending.delete(m.id);
        if (m.type === 'error') p.reject(new Error(m.message)); else p.resolve(m);
      };
      this.worker.onerror = (e) => { this._readyReject && this._readyReject(e); };
    }
    init(modelUrl, preferWebGPU = true, matchModelUrl = '/models/mnn_match.onnx') {
      return new Promise((resolve, reject) => {
        this._ready = resolve; this._readyReject = reject;
        this.worker.postMessage({ type: 'init', modelUrl, matchModelUrl, preferWebGPU });
      });
    }
    /** Mutual-NN matching in the worker (GPU matmul when available). Descriptors are copied, not transferred. */
    match(A, B, minCos) {
      const id = this.nextId++;
      return new Promise((resolve, reject) => {
        this.pending.set(id, { resolve, reject });
        this.worker.postMessage({ type: 'match', id, descA: A.desc, nA: A.count, descB: B.desc, nB: B.count, minCos });
      });
    }
    /** gray: Float32Array (W*H), transferred to the worker. */
    extract(gray, W, H, opts = {}) {
      const id = this.nextId++;
      return new Promise((resolve, reject) => {
        this.pending.set(id, { resolve, reject });
        this.worker.postMessage({ type: 'extract', id, tag: opts.tag, gray, W, H, topK: opts.topK, threshold: opts.threshold }, [gray.buffer]);
      });
    }
  }

  /** Size that fits `maxSide` and is a multiple of 32, keeping aspect. */
  function fitSize(w, h, maxSide) {
    const s = Math.min(1, maxSide / Math.max(w, h));
    return { W: Math.max(32, Math.floor(w * s / 32) * 32), H: Math.max(32, Math.floor(h * s / 32) * 32) };
  }

  /**
   * Draw `source` (canvas/video/image/bitmap) into a scratch canvas of size WxH and
   * return {gray, sx, sy} where sx, sy map scratch pixels back to source pixels.
   */
  function grabGray(source, srcW, srcH, W, H, scratch) {
    scratch.width = W; scratch.height = H;
    const ctx = scratch.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(source, 0, 0, srcW, srcH, 0, 0, W, H);
    const img = ctx.getImageData(0, 0, W, H);
    return { gray: XFeatPost.toGray(img.data, W, H), sx: srcW / W, sy: srcH / H };
  }

  /** Scale worker output (kpts in network-input px) into the caller's frame. Returns a feature set. */
  function toFrame(feat, sx, sy) {
    const kpts = new Float32Array(feat.count * 2);
    for (let i = 0; i < feat.count; i++) { kpts[i * 2] = feat.kpts[i * 2] * sx; kpts[i * 2 + 1] = feat.kpts[i * 2 + 1] * sy; }
    return { kpts, desc: feat.desc, count: feat.count };
  }

  /** Concatenate feature sets that live in the same frame (e.g. a scale pyramid of the screen). */
  function concat(sets) {
    let n = 0;
    for (const s of sets) n += s.count;
    const kpts = new Float32Array(n * 2), desc = new Float32Array(n * 64);
    let o = 0;
    for (const s of sets) {
      kpts.set(s.kpts.subarray(0, s.count * 2), o * 2);
      desc.set(s.desc.subarray(0, s.count * 64), o * 64);
      o += s.count;
    }
    return { kpts, desc, count: n };
  }

  /**
   * Estimate H mapping points of feature set A to feature set B (async; pass opts.client to match in the worker).
   * Both sets: {kpts, desc, count} with kpts already in the frames the caller wants.
   */
  async function estimate(A, B, opts = {}) {
    const pairs = opts.client ? (await opts.client.match(A, B, opts.minCos)).pairs
                              : XFeatPost.match(A.desc, A.count, B.desc, B.count, opts.minCos);
    const n = pairs.length / 2;
    const src = new Float32Array(n * 2), dst = new Float32Array(n * 2);
    for (let k = 0; k < n; k++) {
      const i = pairs[k * 2], j = pairs[k * 2 + 1];
      src[k * 2] = A.kpts[i * 2]; src[k * 2 + 1] = A.kpts[i * 2 + 1];
      dst[k * 2] = B.kpts[j * 2]; dst[k * 2 + 1] = B.kpts[j * 2 + 1];
    }
    const r = n >= 4 ? Homography.ransac(src, dst, n, { threshold: opts.threshold || 4, maxIter: opts.maxIter || 800 }) : null;
    return { nMatches: n, src, dst, H: r ? r.H : null, inliers: r ? r.inliers : new Int32Array(0), nInliers: r ? r.nInliers : 0 };
  }

  /** Reject homographies that fold, flip or blow up the quad (w,h = source frame size). */
  function quadOf(H, w, h) {
    return [Homography.apply(H, 0, 0), Homography.apply(H, w, 0), Homography.apply(H, w, h), Homography.apply(H, 0, h)];
  }
  function quadIsSane(q, limit) {
    // convex, counter-clockwise (same orientation as the source), finite, not absurdly large
    let sign = 0;
    for (let i = 0; i < 4; i++) {
      const a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4];
      if (!isFinite(a[0]) || !isFinite(a[1]) || Math.abs(a[0]) > limit || Math.abs(a[1]) > limit) return false;
      const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
      if (cross === 0) return false;
      const s = Math.sign(cross);
      if (sign === 0) sign = s; else if (s !== sign) return false;
    }
    return sign > 0;
  }

  root.Matcher = { FeatureClient, fitSize, grabGray, toFrame, concat, estimate, quadOf, quadIsSane };
})(typeof self !== 'undefined' ? self : this);
