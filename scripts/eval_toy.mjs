import { genEpisodes, ARENA, step } from '../src/lib/toyworld.js';
import { rollout, K } from '../src/lib/worldmodel.js';
import { unpackMLP } from '../src/lib/nn.js';
import fs from 'fs';
const load = (m) => unpackMLP(JSON.parse(fs.readFileSync(`public/models/toy/${m}.json`)));
const TF = load('teacher'), DF = load('diffusion');
const test = genEpisodes(40, 600, 4242);
const H = [5, 20, 50, 100, 200, 400, 590];
function ev(m, noise, drop2 = false) {
  const err = Object.fromEntries(H.map((h) => [h, 0])); let oob = 0, n = 0, spd = 0, d2b = 0;
  for (const ep of test) {
    const acts = ep.A.slice(K - 1).map(([a, b]) => [a, drop2 ? null : b]);
    const r = rollout(m, ep.S.slice(0, K), acts, noise, 1);
    for (const h of H) { const s = r[h - 1], g = ep.S[K - 1 + h]; err[h] += Math.hypot(s[8] - g[8], s[9] - g[9]); }
    for (const s of r) { n++; if (!(Math.abs(s[8]) < ARENA.w + 0.1 && Math.abs(s[9]) < ARENA.h + 0.1)) oob++; spd += Math.hypot(s[10], s[11]); d2b += Math.hypot(s[4] - s[8], s[5] - s[9]); }
  }
  for (const h of H) err[h] = +(err[h] / test.length).toFixed(3);
  return { err, oob: +(100 * oob / n).toFixed(2), ballSpeed: +(spd / n).toFixed(4), car2ball: +(d2b / n).toFixed(3) };
}
let ts = 0, tn = 0, td = 0; for (const ep of test) for (const s of ep.S) { ts += Math.hypot(s[10], s[11]); td += Math.hypot(s[4] - s[8], s[5] - s[9]); tn++; }
console.log('true ballSpeed', (ts / tn).toFixed(4), 'car2ball', (td / tn).toFixed(3));
console.log('TF', JSON.stringify(ev(TF, 0)));
for (const nz of [0, 0.1, 0.2, 0.3]) console.log('DF', nz, JSON.stringify(ev(DF, nz)));
console.log('DF drop2 0.1', JSON.stringify(ev(DF, 0.1, true)));
// idle car2 baseline in the true engine
let dd = 0, nn = 0; for (const ep of test) { let s = ep.S[0]; for (const [a] of ep.A) { s = step(s, a, 0); dd += Math.hypot(s[4] - s[8], s[5] - s[9]); nn++; } } console.log('idle car2 dist', (dd / nn).toFixed(3));
