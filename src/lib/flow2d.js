// 2-D flow matching + progressive self-distillation (shortcut) — the same objectives MIRA uses (Sec. 4.3),
// on a 2-D toy distribution so you can watch it train live.
import { MLP, mulberry32, randn } from './nn.js';

// Sample target points from text rendered on an offscreen canvas.
export function textDistribution(text = 'MIRA', n = 4000, seed = 1) {
  const W = 400, H = 140;
  const c = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(W, H) : Object.assign(document.createElement('canvas'), { width: W, height: H });
  const g = c.getContext('2d');
  g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
  g.fillStyle = '#fff'; g.font = '900 118px Inter, Arial Black, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillText(text, W / 2, H / 2 + 6);
  const d = g.getImageData(0, 0, W, H).data, on = [];
  for (let y = 0; y < H; y += 1) for (let x = 0; x < W; x += 1) if (d[(y * W + x) * 4] > 128) on.push([x, y]);
  const rng = mulberry32(seed), pts = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    const [x, y] = on[Math.floor(rng() * on.length)];
    pts[2 * i] = ((x + rng() - W / 2) / W) * 4.4; pts[2 * i + 1] = (-(y + rng() - H / 2) / W) * 4.4;
  }
  return pts;
}

// v_theta(z, tau, delta): input [x, y, tau, delta, sin/cos features], output velocity (2)
const IN = 8;
function feat(X, o, x, y, t, d) {
  X[o] = x; X[o + 1] = y; X[o + 2] = t; X[o + 3] = d * 4;
  X[o + 4] = Math.sin(Math.PI * t); X[o + 5] = Math.cos(Math.PI * t); X[o + 6] = Math.sin(2 * Math.PI * t); X[o + 7] = Math.cos(2 * Math.PI * t);
}
export function makeFlowNet(seed = 2) { return new MLP([IN, 128, 128, 128, 2], seed); }

export class FlowTrainer {
  constructor(net, data, seed = 4) { this.net = net; this.data = data; this.n = data.length / 2; this.rng = mulberry32(seed); this.steps = 0; this.loss = null; }
  sampleData(rng) { const i = Math.floor(rng() * this.n); return [this.data[2 * i], this.data[2 * i + 1]]; }

  // Standard flow matching: z_tau = tau*z1 + (1-tau)*z0, regress v -> (z1 - z0). (delta = 0)
  fmStep(B = 256, lr = 2e-3) {
    const X = new Float32Array(B * IN), Y = new Float32Array(B * 2), r = this.rng;
    for (let b = 0; b < B; b++) {
      const [x1, y1] = this.sampleData(r), x0 = randn(r), y0 = randn(r), t = r();
      feat(X, b * IN, t * x1 + (1 - t) * x0, t * y1 + (1 - t) * y0, t, 0);
      Y[2 * b] = x1 - x0; Y[2 * b + 1] = y1 - y0;
    }
    return this.track(this.net.trainMSE(X, Y, B, lr));
  }

  // Progressive self-distillation / shortcut: one big step of size d must equal two half steps (stop-grad target).
  psdStep(B = 256, lr = 1e-3, fracFM = 0.5) {
    const net = this.net, r = this.rng;
    const nFM = Math.floor(B * fracFM), nSC = B - nFM;
    const X = new Float32Array(B * IN), Y = new Float32Array(B * 2);
    for (let b = 0; b < nFM; b++) {
      const [x1, y1] = this.sampleData(r), x0 = randn(r), y0 = randn(r), t = r();
      feat(X, b * IN, t * x1 + (1 - t) * x0, t * y1 + (1 - t) * y0, t, 0);
      Y[2 * b] = x1 - x0; Y[2 * b + 1] = y1 - y0;
    }
    // shortcut targets
    const Xa = new Float32Array(nSC * IN), meta = [];
    for (let j = 0; j < nSC; j++) {
      const k = Math.floor(r() * 6); const d = 1 / Math.pow(2, k); // big step size in {1, 1/2, ..., 1/32}
      const nSlots = Math.round(1 / d); const s = Math.floor(r() * nSlots) * d;
      const [x1, y1] = this.sampleData(r), x0 = randn(r), y0 = randn(r);
      const zx = s * x1 + (1 - s) * x0, zy = s * y1 + (1 - s) * y0;
      const h = d / 2;
      feat(Xa, j * IN, zx, zy, s, h); meta.push({ zx, zy, s, d, h });
    }
    const va = net.forward(Xa, nSC);
    const Xb = new Float32Array(nSC * IN);
    meta.forEach((m, j) => { m.ux = m.zx + m.h * va[2 * j]; m.uy = m.zy + m.h * va[2 * j + 1]; feat(Xb, j * IN, m.ux, m.uy, m.s + m.h, m.h); });
    const vb = net.forward(Xb, nSC);
    meta.forEach((m, j) => {
      const b = nFM + j; feat(X, b * IN, m.zx, m.zy, m.s, m.d);
      Y[2 * b] = 0.5 * (va[2 * j] + vb[2 * j]); Y[2 * b + 1] = 0.5 * (va[2 * j + 1] + vb[2 * j + 1]);
    });
    return this.track(net.trainMSE(X, Y, B, lr));
  }
  track(l) { this.steps++; this.loss = this.loss === null ? l : 0.97 * this.loss + 0.03 * l; return this.loss; }
}

// Euler sampling with N steps. useDelta: tell the network the step size (only meaningful for the shortcut model).
export function sample(net, z0, N, useDelta, keepPath = false) {
  const n = z0.length / 2, z = Float32Array.from(z0), d = 1 / N;
  const paths = keepPath ? [Float32Array.from(z)] : null;
  const X = new Float32Array(n * IN);
  for (let k = 0; k < N; k++) {
    const t = k * d;
    for (let i = 0; i < n; i++) feat(X, i * IN, z[2 * i], z[2 * i + 1], t, useDelta ? d : 0);
    const v = net.forward(X, n);
    for (let i = 0; i < 2 * n; i++) z[i] += d * v[i];
    if (keepPath) paths.push(Float32Array.from(z));
  }
  return keepPath ? paths : z;
}
export function velocityAt(net, pts, t, delta = 0) {
  const n = pts.length / 2, X = new Float32Array(n * IN);
  for (let i = 0; i < n; i++) feat(X, i * IN, pts[2 * i], pts[2 * i + 1], t, delta);
  return net.forward(X, n);
}
export function gaussian(n, seed = 9) { const r = mulberry32(seed), z = new Float32Array(2 * n); for (let i = 0; i < 2 * n; i++) z[i] = randn(r); return z; }
