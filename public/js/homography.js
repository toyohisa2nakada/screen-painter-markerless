// Homography estimation: normalised DLT + RANSAC + inlier refinement. No dependencies.
// H maps points from image A (src) to image B (dst): [x',y',1]^T ~ H [x,y,1]^T (row-major 3x3).
(function (root) {
  'use strict';

  function apply(H, x, y) {
    const w = H[6] * x + H[7] * y + H[8];
    return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
  }

  function mul(A, B) {
    const C = new Float64Array(9);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
    }
    return C;
  }

  function invert(H) {
    const [a, b, c, d, e, f, g, h, i] = H;
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-12) return null;
    const inv = new Float64Array([
      A, -(b * i - c * h), b * f - c * e,
      B, a * i - c * g, -(a * f - c * d),
      C, -(a * h - b * g), a * e - b * d]);
    for (let k = 0; k < 9; k++) inv[k] /= det;
    return inv;
  }

  function normalize(H) {
    const s = H[8] !== 0 ? 1 / H[8] : 1;
    const out = new Float64Array(9);
    for (let k = 0; k < 9; k++) out[k] = H[k] * s;
    return out;
  }

  // Smallest-eigenvector of symmetric 9x9 matrix via cyclic Jacobi.
  function smallestEigvec9(M) {
    const n = 9;
    const A = Float64Array.from(M);
    const V = new Float64Array(n * n);
    for (let i = 0; i < n; i++) V[i * n + i] = 1;
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
    let best = 0;
    for (let i = 1; i < n; i++) if (A[i * n + i] < A[best * n + best]) best = i;
    const v = new Float64Array(n);
    for (let k = 0; k < n; k++) v[k] = V[k * n + best];
    return v;
  }

  // Similarity normalisation (Hartley): returns T and normalised points.
  function normPoints(pts, idx) {
    const n = idx.length;
    let mx = 0, my = 0;
    for (let k = 0; k < n; k++) { mx += pts[idx[k] * 2]; my += pts[idx[k] * 2 + 1]; }
    mx /= n; my /= n;
    let d = 0;
    for (let k = 0; k < n; k++) d += Math.hypot(pts[idx[k] * 2] - mx, pts[idx[k] * 2 + 1] - my);
    d /= n;
    const s = d > 1e-9 ? Math.SQRT2 / d : 1;
    const T = new Float64Array([s, 0, -s * mx, 0, s, -s * my, 0, 0, 1]);
    const out = new Float64Array(n * 2);
    for (let k = 0; k < n; k++) {
      out[k * 2] = s * (pts[idx[k] * 2] - mx);
      out[k * 2 + 1] = s * (pts[idx[k] * 2 + 1] - my);
    }
    return { T, pts: out };
  }

  /** DLT on the given index subset. src/dst: Float32Array|Float64Array of [x,y,...]. */
  function dlt(src, dst, idx) {
    const n = idx.length;
    if (n < 4) return null;
    const a = normPoints(src, idx), b = normPoints(dst, idx);
    const M = new Float64Array(81);
    const r1 = new Float64Array(9), r2 = new Float64Array(9);
    for (let k = 0; k < n; k++) {
      const x = a.pts[k * 2], y = a.pts[k * 2 + 1], u = b.pts[k * 2], v = b.pts[k * 2 + 1];
      r1.set([-x, -y, -1, 0, 0, 0, u * x, u * y, u]);
      r2.set([0, 0, 0, -x, -y, -1, v * x, v * y, v]);
      for (let i = 0; i < 9; i++) for (let j = i; j < 9; j++) {
        M[i * 9 + j] += r1[i] * r1[j] + r2[i] * r2[j];
      }
    }
    for (let i = 0; i < 9; i++) for (let j = 0; j < i; j++) M[i * 9 + j] = M[j * 9 + i];
    const h = smallestEigvec9(M);
    const Tbinv = invert(b.T);
    if (!Tbinv) return null;
    return normalize(mul(mul(Tbinv, h), a.T));
  }

  // Squared reprojection error measured in the *source* frame (dst back-projected through Hinv).
  // The source is the phone image, so the threshold is in phone pixels regardless of how large
  // the screen appears.
  function symErr(H, Hinv, src, dst, i) {
    const x = src[i * 2], y = src[i * 2 + 1], u = dst[i * 2], v = dst[i * 2 + 1];
    const q = apply(Hinv, u, v);
    return (q[0] - x) ** 2 + (q[1] - y) ** 2;
  }

  function isSane(H) {
    if (!H) return false;
    for (let k = 0; k < 9; k++) if (!isFinite(H[k])) return false;
    const det = H[0] * H[4] - H[1] * H[3];
    return Math.abs(det) > 1e-6; // orientation-preserving is checked by the caller on the quad
  }

  /**
   * RANSAC homography.
   * @param src Float32Array [x,y,...] (image A)
   * @param dst Float32Array [x,y,...] (image B)
   * @param n   number of correspondences
   * @returns {H, inliers: Int32Array, nInliers} or null
   */
  function ransac(src, dst, n, opts) {
    opts = opts || {};
    const thr = opts.threshold != null ? opts.threshold : 3.0; // px in the source frame
    const thr2 = thr * thr;
    const maxIter = opts.maxIter || 1000, conf = opts.confidence || 0.995;
    if (n < 4) return null;
    let best = null, bestCount = 0, iters = maxIter;
    const pick = new Int32Array(4);
    const all = new Int32Array(n);
    for (let i = 0; i < n; i++) all[i] = i;
    for (let it = 0; it < iters; it++) {
      // sample 4 distinct
      for (let k = 0; k < 4; k++) {
        let c, ok;
        do { c = (Math.random() * n) | 0; ok = true; for (let m = 0; m < k; m++) if (pick[m] === c) ok = false; } while (!ok);
        pick[k] = c;
      }
      const H = dlt(src, dst, pick);
      if (!isSane(H)) continue;
      const Hi = invert(H);
      if (!Hi) continue;
      let cnt = 0;
      for (let i = 0; i < n; i++) if (symErr(H, Hi, src, dst, i) < thr2) cnt++;
      if (cnt > bestCount) {
        bestCount = cnt; best = H;
        const w = cnt / n;
        const denom = Math.log(1 - Math.pow(w, 4));
        if (denom < 0) iters = Math.min(maxIter, Math.ceil(Math.log(1 - conf) / denom));
      }
    }
    if (!best || bestCount < 4) return null;
    // refine on inliers (2 rounds)
    let H = best;
    for (let round = 0; round < 2; round++) {
      const Hi = invert(H);
      const inl = [];
      for (let i = 0; i < n; i++) if (symErr(H, Hi, src, dst, i) < thr2) inl.push(i);
      if (inl.length < 4) break;
      const Hr = dlt(src, dst, Int32Array.from(inl));
      if (isSane(Hr)) H = Hr; else break;
    }
    const Hi = invert(H);
    const inliers = [];
    for (let i = 0; i < n; i++) if (symErr(H, Hi, src, dst, i) < thr2) inliers.push(i);
    return { H, inliers: Int32Array.from(inliers), nInliers: inliers.length };
  }

  /** Homography from exactly/at least 4 point pairs, no RANSAC. */
  function fromPoints(src, dst) {
    const idx = new Int32Array(src.length / 2);
    for (let i = 0; i < idx.length; i++) idx[i] = i;
    return dlt(src, dst, idx);
  }

  root.Homography = { apply, mul, invert, normalize, dlt, ransac, fromPoints };
})(typeof self !== 'undefined' ? self : this);
