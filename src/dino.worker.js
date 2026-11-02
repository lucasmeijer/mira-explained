// Runs the real DINOv3 ViT-S/16 (same family as MIRA's frozen DINOv3-L/16 encoder) in a Web Worker.
let ort, session, provider;
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

async function init(variant, pref) {
  // Dev: local copies. Production (static hosting with per-file size limits): official CDNs.
  const ORT_BASE = import.meta.env.DEV ? self.location.origin + '/ort/' : 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
  ort = await import(/* @vite-ignore */ ORT_BASE + 'ort.all.min.mjs');
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  const base = import.meta.env.DEV ? self.location.origin + '/models/dinov3-s16/' : 'https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX/resolve/main/onnx/';
  const file = variant === 'q8' ? 'model_quantized' : 'model';
  const [onnx, data] = await Promise.all([fetchWithProgress(base + file + '.onnx'), fetchWithProgress(base + file + '.onnx_data', true)]);
  const opts = (ep) => ({ executionProviders: [ep], externalData: [{ path: file + '.onnx_data', data }], graphOptimizationLevel: 'all' });
  const eps = [];
  if (self.navigator.gpu && pref !== 'wasm') eps.push('webgpu');
  if (pref !== 'webgpu' || !eps.length) eps.push('wasm');
  let lastErr;
  for (const ep of eps) {
    try { session = await ort.InferenceSession.create(onnx, opts(ep)); provider = ep; break; }
    catch (e) { lastErr = e; }
  }
  if (!session) throw lastErr;
  return provider;
}

async function fetchWithProgress(url, report) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('fetch failed ' + url);
  const total = +r.headers.get('content-length') || 0;
  if (!report || !r.body || !total) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader(); const buf = new Uint8Array(total); let got = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    buf.set(value, got); got += value.length;
    self.postMessage({ type: 'progress', loaded: got, total });
  }
  return buf;
}

async function embed(rgba, W, H) {
  const N = W * H, x = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) for (let c = 0; c < 3; c++) x[c * N + i] = (rgba[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  const t0 = performance.now();
  const out = await session.run({ pixel_values: new ort.Tensor('float32', x, [1, 3, H, W]) });
  const ms = performance.now() - t0;
  const h = out.last_hidden_state; const D = h.dims[2];
  const data = h.data instanceof Float32Array ? h.data : Float32Array.from(await h.getData());
  // drop CLS + 4 register tokens -> patch tokens only
  const feats = data.slice(5 * D);
  return { feats, D, gh: H / 16, gw: W / 16, ms };
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') { const p = await init(m.variant, m.pref); self.postMessage({ type: 'ready', provider: p, id: m.id }); }
    else if (m.type === 'embed') {
      const r = await embed(m.rgba, m.W, m.H);
      self.postMessage({ type: 'feats', id: m.id, ...r }, [r.feats.buffer]);
    }
  } catch (err) { self.postMessage({ type: 'error', id: m.id, message: String(err && err.message || err) }); }
};
