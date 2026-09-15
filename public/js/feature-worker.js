// Worker: runs the XFeat ONNX backbone with onnxruntime-web (WebGPU, falling back to WASM)
// and the JS post-processing. Messages:
//   {type:'init', modelUrl, matchModelUrl, preferWebGPU}  -> {type:'ready', backend}
//   {type:'match', id, descA, nA, descB, nB, minCos} -> {type:'matches', id, pairs}
//   {type:'extract', id, tag, gray:Float32Array, W, H, topK, threshold} -> {type:'features', ...}
'use strict';
importScripts('/vendor/ort/ort.min.js');
importScripts('/js/xfeat.js');

let session = null;
let matchSession = null;
let backend = null;

// 二重実行を防ぐためのフラグ
let isExtractBusy = false;
let isMatchBusy = false;

async function init(msg) {
  ort.env.wasm.wasmPaths = '/vendor/ort/';
  ort.env.logLevel = 'warning';
  const tries = [];
  if (msg.preferWebGPU !== false && typeof navigator !== 'undefined' && navigator.gpu) tries.push('webgpu');
  tries.push('wasm');
  let lastErr = null;
  for (const ep of tries) {
    try {
      session = await ort.InferenceSession.create(msg.modelUrl, {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
      });
      backend = ep;
      // warm-up
      const W = 640, H = 480;
      await session.run({ image: new ort.Tensor('float32', new Float32Array(W * H), [1, 1, H, W]) });
      if (msg.matchModelUrl) {
        try {
          matchSession = await ort.InferenceSession.create(msg.matchModelUrl, { executionProviders: [ep] });
          await matchSession.run({ a: new ort.Tensor('float32', new Float32Array(8 * 64), [8, 64]), b: new ort.Tensor('float32', new Float32Array(8 * 64), [8, 64]) });
        } catch (e) { console.warn('[feature-worker] match model unavailable, using JS matcher:', e && e.message); matchSession = null; }
      }
      return;
    } catch (e) {
      lastErr = e;
      console.warn('[feature-worker] EP failed:', ep, e && e.message);
      session = null;
    }
  }
  throw lastErr || new Error('no execution provider available');
}

async function extract(msg) {
  if (isExtractBusy) return; // 抽出処理中の場合はスキップ
  isExtractBusy = true;
  try {
    const { id, tag, gray, W, H } = msg;
    const t0 = performance.now();
    const input = new ort.Tensor('float32', gray, [1, 1, H, W]);
    const out = await session.run({ image: input });
    const t1 = performance.now();
    const res = XFeatPost.extract(out.feats.data, out.heatmap.data, out.reliab.data, H, W, {
      topK: msg.topK || 1024, threshold: msg.threshold != null ? msg.threshold : 0.05,
    });
    const t2 = performance.now();
    self.postMessage({
      type: 'features', id, tag, W, H,
      kpts: res.kpts, scores: res.scores, desc: res.desc, count: res.count,
      msNet: t1 - t0, msPost: t2 - t1, backend,
    }, [res.kpts.buffer, res.scores.buffer, res.desc.buffer]);
  } finally {
    isExtractBusy = false;
  }
}

// Mutual nearest neighbour matching: descA (nA x 64), descB (nB x 64) -> Int32Array pairs
async function matchDesc(msg) {
  if (isMatchBusy) return; // 処理中の場合はスキップ
  isMatchBusy = true;
  try {
    const { id, descA, nA, descB, nB } = msg;
    const minCos = msg.minCos != null ? msg.minCos : 0.82;
    const t0 = performance.now();
    let pairs;
    if (matchSession && nA > 0 && nB > 0) {
      const out = await matchSession.run({
        a: new ort.Tensor('float32', descA.subarray(0, nA * 64), [nA, 64]),
        b: new ort.Tensor('float32', descB.subarray(0, nB * 64), [nB, 64]),
      });
      const S = out.s.data; // nA x nB cosine similarities; arg-max in JS (cheap, and avoids EP quirks)
      const bestB = new Int32Array(nA), bestBv = new Float32Array(nA).fill(-2);
      const bestA = new Int32Array(nB), bestAv = new Float32Array(nB).fill(-2);
      for (let i = 0; i < nA; i++) {
        const o = i * nB;
        for (let j = 0; j < nB; j++) {
          const v = S[o + j];
          if (v > bestBv[i]) { bestBv[i] = v; bestB[i] = j; }
          if (v > bestAv[j]) { bestAv[j] = v; bestA[j] = i; }
        }
      }
      const res = [];
      for (let i = 0; i < nA; i++) {
        const j = bestB[i];
        if (bestA[j] === i && bestBv[i] > minCos) res.push(i, j);
      }
      pairs = Int32Array.from(res);
    } else {
      pairs = XFeatPost.match(descA, nA, descB, nB, minCos);
    }
    self.postMessage({ type: 'matches', id, pairs, ms: performance.now() - t0 }, [pairs.buffer]);
  } finally {
    isMatchBusy = false;
  }
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      await init(msg);
      self.postMessage({ type: 'ready', backend });
    } else if (msg.type === 'extract') {
      if (!session) throw new Error('session not ready');
      await extract(msg);
    } else if (msg.type === 'match') {
      await matchDesc(msg);
    }
  } catch (e) {
    self.postMessage({ type: 'error', id: msg.id, tag: msg.tag, message: String(e && e.stack || e) });
  }
};
