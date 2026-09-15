// XFeat post-processing in plain JS (mirrors modules/xfeat.py detectAndCompute):
//  - NMS on the keypoint heatmap (5x5 window, threshold)
//  - score = heatmap * bilinear(reliability)
//  - top-k
//  - bicubic sampling of the 64-d dense descriptor map, L2 normalised
// plus mutual-nearest-neighbour matching.
// Works both in a Worker (importScripts) and on the main thread.
(function (root) {
  'use strict';

  // grid_sample(align_corners=False) with XFeat's normgrid: pixel = x * w/(W-1) - 0.5
  function toMap(x, W, w) { return x * w / (W - 1) - 0.5; }

  function bilinear1(map, w, h, px, py) {
    const x0 = Math.floor(px), y0 = Math.floor(py);
    const fx = px - x0, fy = py - y0;
    let v = 0;
    for (let j = 0; j < 2; j++) {
      const y = y0 + j; if (y < 0 || y >= h) continue;
      const wy = j ? fy : 1 - fy;
      for (let i = 0; i < 2; i++) {
        const x = x0 + i; if (x < 0 || x >= w) continue;
        const wx = i ? fx : 1 - fx;
        v += map[y * w + x] * wx * wy;
      }
    }
    return v;
  }

  // cubic convolution coefficients (PyTorch bicubic, a = -0.75)
  function cubicCoeffs(t, out) {
    const A = -0.75;
    const x1 = t, x2 = 1 - t;
    out[0] = ((A * (x1 + 1) - 5 * A) * (x1 + 1) + 8 * A) * (x1 + 1) - 4 * A;
    out[1] = ((A + 2) * x1 - (A + 3)) * x1 * x1 + 1;
    out[2] = ((A + 2) * x2 - (A + 3)) * x2 * x2 + 1;
    out[3] = ((A * (x2 + 1) - 5 * A) * (x2 + 1) + 8 * A) * (x2 + 1) - 4 * A;
  }

  /**
   * @param feats  Float32Array (64*h*w), CHW, L2-normalised per pixel
   * @param heat   Float32Array (H*W)
   * @param rel    Float32Array (h*w)
   * @returns {kpts: Float32Array(N*2) [x,y in input pixels], scores: Float32Array(N), desc: Float32Array(N*64)}
   */
  function extract(feats, heat, rel, H, W, opts) {
    opts = opts || {};
    const thr = opts.threshold != null ? opts.threshold : 0.05;
    const topK = opts.topK || 1024;
    const r = (opts.nms || 5) >> 1;
    const h = H >> 3, w = W >> 3;

    // --- NMS + reliability score
    const cand = []; // [score, x, y]
    for (let y = 0; y < H; y++) {
      const row = y * W;
      for (let x = 0; x < W; x++) {
        const v = heat[row + x];
        if (v <= thr) continue;
        let isMax = true;
        const y0 = Math.max(0, y - r), y1 = Math.min(H - 1, y + r);
        const x0 = Math.max(0, x - r), x1 = Math.min(W - 1, x + r);
        for (let yy = y0; yy <= y1 && isMax; yy++) {
          const rr = yy * W;
          for (let xx = x0; xx <= x1; xx++) {
            if (heat[rr + xx] > v) { isMax = false; break; }
          }
        }
        if (!isMax) continue;
        const s = v * bilinear1(rel, w, h, toMap(x, W, w), toMap(y, H, h));
        if (s > 0) cand.push([s, x, y]);
      }
    }
    cand.sort((a, b) => b[0] - a[0]);
    const N = Math.min(topK, cand.length);
    const kpts = new Float32Array(N * 2);
    const scores = new Float32Array(N);
    const desc = new Float32Array(N * 64);

    // --- bicubic descriptor sampling
    const cx = new Float64Array(4), cy = new Float64Array(4);
    const plane = h * w;
    const acc = new Float64Array(64);
    const subpix = opts.subpix !== false;
    for (let n = 0; n < N; n++) {
      const [s, x, y] = cand[n];
      let fx = x, fy = y;
      if (subpix && x > 0 && y > 0 && x < W - 1 && y < H - 1) {
        // quadratic fit on the 3-neighbourhood of the heatmap (per axis)
        const c = heat[y * W + x];
        const l = heat[y * W + x - 1], rgt = heat[y * W + x + 1];
        const u = heat[(y - 1) * W + x], d = heat[(y + 1) * W + x];
        const dx = 0.5 * (rgt - l) / (2 * c - l - rgt || 1e-9);
        const dy = 0.5 * (d - u) / (2 * c - u - d || 1e-9);
        if (Math.abs(dx) < 1) fx += dx;
        if (Math.abs(dy) < 1) fy += dy;
      }
      kpts[n * 2] = fx; kpts[n * 2 + 1] = fy; scores[n] = s;
      const px = toMap(x, W, w), py = toMap(y, H, h);
      const ix = Math.floor(px), iy = Math.floor(py);
      cubicCoeffs(px - ix, cx); cubicCoeffs(py - iy, cy);
      acc.fill(0);
      for (let j = 0; j < 4; j++) {
        const yy = iy - 1 + j; if (yy < 0 || yy >= h) continue;
        for (let i = 0; i < 4; i++) {
          const xx = ix - 1 + i; if (xx < 0 || xx >= w) continue;
          const wgt = cx[i] * cy[j];
          const base = yy * w + xx;
          for (let c = 0; c < 64; c++) acc[c] += wgt * feats[c * plane + base];
        }
      }
      let norm = 0;
      for (let c = 0; c < 64; c++) norm += acc[c] * acc[c];
      norm = norm > 0 ? 1 / Math.sqrt(norm) : 0;
      const o = n * 64;
      for (let c = 0; c < 64; c++) desc[o + c] = acc[c] * norm;
    }
    return { kpts, scores, desc, count: N };
  }

  /**
   * Mutual nearest neighbour matching on L2-normalised descriptors.
   * @returns Int32Array of pairs [i0, i1, i0, i1, ...] (indices into a / b)
   */
  function match(descA, nA, descB, nB, minCos) {
    minCos = minCos != null ? minCos : 0.82;
    if (!nA || !nB) return new Int32Array(0);
    const bestB = new Int32Array(nA), bestBv = new Float32Array(nA).fill(-2);
    const bestA = new Int32Array(nB), bestAv = new Float32Array(nB).fill(-2);
    for (let i = 0; i < nA; i++) {
      const oa = i * 64;
      for (let j = 0; j < nB; j++) {
        const ob = j * 64;
        let s = 0;
        for (let c = 0; c < 64; c += 8) {
          s += descA[oa + c] * descB[ob + c] + descA[oa + c + 1] * descB[ob + c + 1]
             + descA[oa + c + 2] * descB[ob + c + 2] + descA[oa + c + 3] * descB[ob + c + 3]
             + descA[oa + c + 4] * descB[ob + c + 4] + descA[oa + c + 5] * descB[ob + c + 5]
             + descA[oa + c + 6] * descB[ob + c + 6] + descA[oa + c + 7] * descB[ob + c + 7];
        }
        if (s > bestBv[i]) { bestBv[i] = s; bestB[i] = j; }
        if (s > bestAv[j]) { bestAv[j] = s; bestA[j] = i; }
      }
    }
    const out = [];
    for (let i = 0; i < nA; i++) {
      const j = bestB[i];
      if (bestA[j] === i && bestBv[i] > minCos) out.push(i, j);
    }
    return Int32Array.from(out);
  }

  /** Convert RGBA ImageData to a Float32Array grayscale (0..255), NCHW 1x1xHxW. */
  function toGray(rgba, W, H) {
    const g = new Float32Array(W * H);
    for (let i = 0, p = 0; i < g.length; i++, p += 4) {
      g[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
    }
    return g;
  }

  root.XFeatPost = { extract, match, toGray };
})(typeof self !== 'undefined' ? self : this);
