// Tiny dependency-free MLP + Adam, fast enough to train small models live in the browser.
// Row-major Float32Array matrices. Works in browser and Node.

export function randn(rng = Math.random) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Seeded PRNG (mulberry32) so demos are reproducible.
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const silu = (x) => x / (1 + Math.exp(-x));
const dsilu = (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 + x * (1 - s)); };

export class MLP {
  // sizes: [in, h1, h2, ..., out]
  constructor(sizes, seed = 1) {
    this.sizes = sizes;
    const rng = mulberry32(seed);
    this.W = []; this.b = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const fin = sizes[l], fout = sizes[l + 1];
      const W = new Float32Array(fin * fout);
      const scale = Math.sqrt(2 / fin) * (l === sizes.length - 2 ? 0.3 : 1);
      for (let i = 0; i < W.length; i++) W[i] = randn(rng) * scale;
      this.W.push(W); this.b.push(new Float32Array(fout));
    }
    this.initAdam();
  }
  initAdam() {
    this.mW = this.W.map((w) => new Float32Array(w.length));
    this.vW = this.W.map((w) => new Float32Array(w.length));
    this.mb = this.b.map((w) => new Float32Array(w.length));
    this.vb = this.b.map((w) => new Float32Array(w.length));
    this.t = 0;
  }
  get nParams() { return this.W.reduce((s, w) => s + w.length, 0) + this.b.reduce((s, w) => s + w.length, 0); }

  // X: Float32Array (B*in). Returns output (B*out). If keep, stores activations for backward.
  forward(X, B, keep = false) {
    let h = X; const L = this.W.length;
    if (keep) { this.pre = []; this.acts = [X]; this.B = B; }
    for (let l = 0; l < L; l++) {
      const fin = this.sizes[l], fout = this.sizes[l + 1], W = this.W[l], bias = this.b[l];
      const z = new Float32Array(B * fout);
      for (let n = 0; n < B; n++) {
        const zo = n * fout, ho = n * fin;
        for (let j = 0; j < fout; j++) z[zo + j] = bias[j];
        for (let i = 0; i < fin; i++) {
          const hv = h[ho + i]; if (hv === 0) continue;
          const wo = i * fout;
          for (let j = 0; j < fout; j++) z[zo + j] += hv * W[wo + j];
        }
      }
      if (l < L - 1) {
        const a = new Float32Array(z.length);
        for (let k = 0; k < z.length; k++) a[k] = silu(z[k]);
        if (keep) { this.pre.push(z); this.acts.push(a); }
        h = a;
      } else {
        if (keep) this.pre.push(z);
        h = z;
      }
    }
    return h;
  }

  // dOut: gradient wrt output (B*out). Accumulates grads, returns nothing.
  backward(dOut) {
    const B = this.B, L = this.W.length;
    this.gW = this.W.map((w) => new Float32Array(w.length));
    this.gb = this.b.map((w) => new Float32Array(w.length));
    let d = dOut;
    for (let l = L - 1; l >= 0; l--) {
      const fin = this.sizes[l], fout = this.sizes[l + 1], W = this.W[l];
      const a = this.acts[l], gW = this.gW[l], gb = this.gb[l];
      const dPrev = l > 0 ? new Float32Array(B * fin) : null;
      for (let n = 0; n < B; n++) {
        const doff = n * fout, aoff = n * fin;
        for (let j = 0; j < fout; j++) gb[j] += d[doff + j];
        for (let i = 0; i < fin; i++) {
          const av = a[aoff + i]; const wo = i * fout; let s = 0;
          for (let j = 0; j < fout; j++) { const dj = d[doff + j]; gW[wo + j] += av * dj; s += W[wo + j] * dj; }
          if (dPrev) dPrev[aoff + i] = s;
        }
      }
      if (l > 0) { const z = this.pre[l - 1]; for (let k = 0; k < dPrev.length; k++) dPrev[k] *= dsilu(z[k]); d = dPrev; }
    }
  }

  step(lr = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8, clip = 1.0) {
    this.t++;
    // global grad-norm clipping
    let gn = 0;
    for (const g of this.gW) for (let i = 0; i < g.length; i++) gn += g[i] * g[i];
    for (const g of this.gb) for (let i = 0; i < g.length; i++) gn += g[i] * g[i];
    gn = Math.sqrt(gn); const sc = gn > clip ? clip / gn : 1;
    const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
    const upd = (P, G, M, V) => {
      for (let i = 0; i < P.length; i++) {
        const g = G[i] * sc;
        M[i] = b1 * M[i] + (1 - b1) * g; V[i] = b2 * V[i] + (1 - b2) * g * g;
        P[i] -= lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + eps);
      }
    };
    for (let l = 0; l < this.W.length; l++) { upd(this.W[l], this.gW[l], this.mW[l], this.vW[l]); upd(this.b[l], this.gb[l], this.mb[l], this.vb[l]); }
  }

  // MSE training step. Y target (B*out). Returns loss.
  trainMSE(X, Y, B, lr) {
    const out = this.forward(X, B, true);
    const d = new Float32Array(out.length); let loss = 0;
    for (let k = 0; k < out.length; k++) { const e = out[k] - Y[k]; loss += e * e; d[k] = (2 * e) / out.length; }
    this.backward(d); this.step(lr);
    return loss / out.length;
  }

  toJSON() { return { sizes: this.sizes, W: this.W.map((w) => Array.from(w, (v) => +v.toFixed(5))), b: this.b.map((w) => Array.from(w, (v) => +v.toFixed(5))) }; }
  static fromJSON(j) { const m = new MLP(j.sizes); m.W = j.W.map((w) => Float32Array.from(w)); m.b = j.b.map((w) => Float32Array.from(w)); m.initAdam(); return m; }
}

// compact binary serialisation (base64 float32)
function f32ToB64(arr) {
  const u8 = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength); let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return typeof btoa === 'function' ? btoa(s) : Buffer.from(u8).toString('base64');
}
function b64ToF32(b) {
  const bin = typeof atob === 'function' ? atob(b) : Buffer.from(b, 'base64').toString('binary');
  const u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}
export function packMLP(m, extra = {}) { return { sizes: m.sizes, W: m.W.map(f32ToB64), b: m.b.map(f32ToB64), ...extra }; }
export function unpackMLP(j) { const m = new MLP(j.sizes); m.W = j.W.map(b64ToF32); m.b = j.b.map(b64ToF32); m.initAdam(); if (j.residual) m.residual = j.residual; return m; }
