// A tiny learned world model for Rocket Pong.
// p(next state | last K states, both players' actions)  — the same factorisation MIRA uses,
// with a 12-number state standing in for MIRA's 9x16x32 latent frame.
import { MLP, mulberry32, randn } from './nn.js';
import { STATE_DIM, N_ACT, norm, denorm } from './toyworld.js';

export const K = 4; // context window (MIRA: 20 latent frames at inference)
const ACT_IN = 2 * (N_ACT + 1); // one-hot + "absent" token per player
export const IN_DIM = K * STATE_DIM + K + ACT_IN;

export function makeModel(seed = 3, hidden = [192, 192, 192]) {
  return new MLP([IN_DIM, ...hidden, STATE_DIM], seed);
}

// ctx: array of K normalised states (oldest first), sig: K noise levels, acts: [a1,a2] (null = absent)
function encode(X, off, ctx, sig, acts) {
  let p = off;
  for (let k = 0; k < K; k++) for (let d = 0; d < STATE_DIM; d++) X[p++] = ctx[k][d];
  for (let k = 0; k < K; k++) X[p++] = sig[k] * 3;
  for (let pl = 0; pl < 2; pl++) {
    const a = acts[pl];
    for (let i = 0; i <= N_ACT; i++) X[p++] = 0;
    if (a === null || a === undefined) X[p - 1] = 1; else X[p - 1 - N_ACT + a] = 1;
  }
}

export class Trainer {
  // mode: 'teacher' (clean context) | 'diffusion' (each context frame independently noised)
  constructor(model, episodes, { mode = 'diffusion', sigmaMax = 0.5, sigmaPow = 1.5, residual = 'noisy', actionDropout = 0, seed = 11 } = {}) {
    this.m = model; this.mode = mode; this.sigmaMax = sigmaMax; this.sigmaPow = sigmaPow; this.residual = residual; this.pDrop = actionDropout;
    this.rng = mulberry32(seed);
    this.Z = episodes.map((e) => e.S.map(norm)); this.A = episodes.map((e) => e.A);
    this.steps = 0; this.lossEMA = null;
  }
  batch(B) {
    const X = new Float32Array(B * IN_DIM), Y = new Float32Array(B * STATE_DIM), rng = this.rng;
    for (let n = 0; n < B; n++) {
      const e = Math.floor(rng() * this.Z.length), Z = this.Z[e], A = this.A[e];
      const t = K - 1 + Math.floor(rng() * (A.length - K + 1));
      const ctx = [], sig = [];
      for (let k = 0; k < K; k++) {
        const z = Z[t - K + 1 + k];
        // diffusion forcing: an independent noise level per context frame
        const s = this.mode === 'diffusion' ? this.sigmaMax * Math.pow(rng(), this.sigmaPow) : 0;
        sig.push(s);
        ctx.push(s > 0 ? z.map((v) => v + s * randn(rng)) : z);
      }
      const acts = A[t].map((a) => (rng() < this.pDrop ? null : a));
      encode(X, n * IN_DIM, ctx, sig, acts);
      // The network only *sees* the noised context; its output is a correction added to the clean last frame
      // (at inference that is the model's own previous prediction), so context noise never leaks into the output.
      const last = this.residual === 'clean' ? Z[t] : ctx[K - 1], tgt = Z[t + 1];
      for (let d = 0; d < STATE_DIM; d++) Y[n * STATE_DIM + d] = tgt[d] - last[d];
    }
    return { X, Y };
  }
  train(nSteps = 1, B = 128, lr = 1e-3) {
    let l = 0;
    for (let i = 0; i < nSteps; i++) {
      const { X, Y } = this.batch(B);
      l = this.m.trainMSE(X, Y, B, lr);
      this.lossEMA = this.lossEMA === null ? l : 0.98 * this.lossEMA + 0.02 * l;
      this.steps++;
    }
    return this.lossEMA;
  }
}

// Predict the next raw state from a context of K raw states.
// ctxNoise: noise std re-injected into the (self-generated) context at inference (paper default: 0.2)
export function predictNext(model, ctxRaw, acts, ctxNoise = 0, rng = Math.random, residual = model.residual || 'noisy') {
  const clean = ctxRaw.map(norm);
  const ctx = clean.map((z) => (ctxNoise > 0 ? z.map((v) => v + ctxNoise * randn(rng)) : z));
  const X = new Float32Array(IN_DIM);
  encode(X, 0, ctx, new Array(K).fill(ctxNoise), acts);
  const d = model.forward(X, 1);
  const last = residual === 'clean' ? clean[K - 1] : ctx[K - 1]; const z = new Float32Array(STATE_DIM);
  for (let i = 0; i < STATE_DIM; i++) z[i] = last[i] + d[i];
  return denorm(z);
}

// Autoregressive rollout: the model's own outputs become its future context.
export function rollout(model, initCtx, actionSeq, ctxNoise = 0, seed = 5) {
  const rng = mulberry32(seed);
  const ctx = initCtx.slice(-K); const out = [];
  for (const acts of actionSeq) {
    const s = predictNext(model, ctx, acts, ctxNoise, rng);
    out.push(s); ctx.push(s); ctx.shift();
  }
  return out;
}
