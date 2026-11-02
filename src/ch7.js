import { mountNav, $ } from './nav.js';
import { genEpisodes, norm, N_ACT } from './lib/toyworld.js';
import { rollout, K } from './lib/worldmodel.js';
import { MLP, unpackMLP, mulberry32 } from './lib/nn.js';
import { barChart } from './lib/charts.js';
mountNav('ch7-evidence');

const load = async (m) => unpackMLP(await (await fetch(`/models/toy/${m}.json`)).json());
const [TF, DF] = await Promise.all([load('teacher'), load('diffusion')]);
const ACTS = ['idle', '↑', '↓', '←', '→'];

// features for the inverse-dynamics probe: P1-car part of state t and the change to t+1
function feat(X, o, s0, s1) {
  const a = norm(s0), b = norm(s1);
  for (let d = 0; d < 4; d++) { X[o + d] = a[d]; X[o + 4 + d] = (b[d] - a[d]) * 8; }
  X[o + 8] = a[8] - a[0]; X[o + 9] = a[9] - a[1];
}
const FD = 10;
function ap(scores, labels) {
  const idx = scores.map((s, i) => i).sort((i, j) => scores[j] - scores[i]);
  let tp = 0, sum = 0, P = labels.filter(Boolean).length; if (!P) return NaN;
  idx.forEach((i, r) => { if (labels[i]) { tp++; sum += tp / (r + 1); } });
  return sum / P;
}
function mAP(probe, pairs) {
  const n = pairs.length, X = new Float32Array(n * FD);
  pairs.forEach((p, i) => feat(X, i * FD, p[0], p[1]));
  const out = probe.forward(X, n);
  const aps = [];
  for (let k = 0; k < N_ACT; k++) { const v = ap(Array.from({ length: n }, (_, i) => out[i * N_ACT + k]), pairs.map((p) => p[2] === k)); if (!isNaN(v)) aps.push(v); }
  return aps.reduce((a, b) => a + b, 0) / aps.length;
}

$('#btnARR').onclick = async () => {
  $('#btnARR').disabled = true;
  const info = (t) => { $('#arrInfo').textContent = t; return new Promise((r) => setTimeout(r, 20)); };
  await info('generating engine data…');
  const train = genEpisodes(150, 200, 777), test = genEpisodes(40, 310, 4243);
  const probe = new MLP([FD, 96, 96, N_ACT], 5), rng = mulberry32(3);
  const pairs = []; for (const e of train) for (let t = 0; t < e.A.length; t++) pairs.push([e.S[t], e.S[t + 1], e.A[t][0]]);
  for (let it = 0; it < 2500; it++) {
    const B = 128, X = new Float32Array(B * FD), Y = new Float32Array(B * N_ACT);
    for (let b = 0; b < B; b++) { const p = pairs[Math.floor(rng() * pairs.length)]; feat(X, b * FD, p[0], p[1]); Y[b * N_ACT + p[2]] = 1; }
    probe.trainMSE(X, Y, B, it < 1800 ? 2e-3 : 5e-4);
    if (it % 250 === 0) await info(`training action probe… ${((it / 2500) * 100) | 0}%`);
  }
  await info('rolling out world models with the real key sequences…');
  const real = [[], []];
  const runs = { dfKeys: [[], []], dfHidden: [[], []], tfKeys: [[], []] };
  const EARLY = 60; // 4 s at 15 Hz = MIRA's training-clip length
  for (const e of test) {
    for (let t = K - 1; t < e.A.length; t++) real[t - K + 1 < EARLY ? 0 : 1].push([e.S[t], e.S[t + 1], e.A[t][0]]);
    const acts = e.A.slice(K - 1);
    const cfgs = [['dfKeys', DF, acts], ['dfHidden', DF, acts.map(([, b]) => [null, b])], ['tfKeys', TF, acts]];
    for (const [name, m, a] of cfgs) {
      const r = rollout(m, e.S.slice(0, K), a, 0, 9);
      const st = [e.S[K - 1], ...r];
      for (let h = 0; h < acts.length; h++) {
        const ok = st[h + 1].every((v) => Number.isFinite(v) && Math.abs(v) < 50);
        // the probe is always scored against the keys that were *commanded*, even when the model wasn't told them
        runs[name][h < EARLY ? 0 : 1].push(ok ? [st[h], st[h + 1], acts[h][0]] : [st[h].map(() => 0), st[h].map(() => 0), acts[h][0]]);
      }
    }
  }
  await info('scoring…');
  const mE = mAP(probe, real[0]), mL = mAP(probe, real[1]);
  const arr = (name, w) => mAP(probe, runs[name][w]) / (w ? mL : mE);
  const rows = [
    { label: 'diffusion · keys given · 0–4 s', value: arr('dfKeys', 0), color: '#fb923c' },
    { label: 'diffusion · keys given · 4–20 s', value: arr('dfKeys', 1), color: '#fb923c' },
    { label: 'teacher · keys given · 0–4 s', value: arr('tfKeys', 0), color: '#38bdf8' },
    { label: 'teacher · keys given · 4–20 s', value: arr('tfKeys', 1), color: '#38bdf8' },
    { label: 'diffusion · keys HIDDEN', value: (arr('dfHidden', 0) + arr('dfHidden', 1)) / 2, color: '#66728a' },
  ];
  barChart($('#chARR'), { title: 'ARR = mAP(generated) / mAP(real)   (1.0 = perfectly obeys keys)', better: 'higher', fmt: (v) => v.toFixed(2), max: 1.1, data: rows, width: 560, labelW: 215 });
  $('#arrInfo').textContent = `probe mAP on real transitions: ${mE.toFixed(2)}`;
  $('#arrNote').innerHTML = `<b>How to read it.</b> The grey bar is the sanity check: when P1's keys are hidden from the model, the keys it was "commanded" are barely recoverable from its
    rollout. That's what an uncontrollable model scores. With the keys given, both toy models clearly obey them. The honest finding: in this <i>toy</i>, teacher forcing is
    actually the more precise one while it lasts, and diffusion forcing pays a little controllability for never diverging (noisy-context training makes the toy slightly sluggish).
    At MIRA's scale, the paper reports no such trade-off: diffusion forcing wins on every metric (Table 8), and ARR reaches 0.91.`;
  $('#btnARR').disabled = false; $('#btnARR').textContent = '↻ Run again';
};
