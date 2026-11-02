import { mountNav, $, $$ } from './nav.js';
import { kickoff, step, botAction, genEpisodes } from './lib/toyworld.js';
import { predictNext, rollout, K } from './lib/worldmodel.js';
import { unpackMLP, mulberry32 } from './lib/nn.js';
import { drawArena, isDiverged } from './lib/toydraw.js';
import { lineChart } from './lib/charts.js';
mountNav('ch5-multiplayer');

const DF = unpackMLP(await (await fetch('/models/toy/diffusion.json')).json());
const rng = mulberry32(99);
const cv = $('#cvPlay'), g = cv.getContext('2d');
const mode = { p1: 'keys', p2: 'auto' };
const held = { p1: [], p2: [] };
const K1 = { w: 1, s: 2, a: 3, d: 4 }, K2 = { arrowup: 1, arrowdown: 2, arrowleft: 3, arrowright: 4 };
let W;

function prime() {
  let s = kickoff(rng); const hist = [s];
  for (let i = 0; i < K - 1; i++) { s = step(s, 0, 0); hist.push(s); }
  W = { ctx: hist.slice(), eng: s, t: 0, trail: [], infl: [[0, 0, 0], [0, 0, 0]], acts: [0, 0] };
}
prime();

function actionFor(p, s) {
  const m = mode[`p${p + 1}`];
  if (m === 'auto') return null;
  if (m === 'idle') return 0;
  const h = held[`p${p + 1}`]; if (!h.length) return 0;
  return (p === 0 ? K1 : K2)[h[h.length - 1]];
}

function tick() {
  const s = W.ctx[W.ctx.length - 1];
  const a = [actionFor(0, s), actionFor(1, s)];
  W.acts = a;
  // attribution probe (deterministic, no context noise)
  const ctx = W.ctx.slice(-K);
  for (let p = 0; p < 2; p++) {
    const outs = [];
    for (let k = 0; k < 5; k++) { const aa = a.slice(); aa[p] = k; outs.push(predictNext(DF, ctx, aa, 0)); }
    const spread = (o) => { let mx = 0, my = 0; outs.forEach((q) => { mx += q[o]; my += q[o + 1]; }); mx /= 5; my /= 5; return outs.reduce((acc, q) => acc + Math.hypot(q[o] - mx, q[o + 1] - my), 0) / 5; };
    [0, 4, 8].forEach((o, j) => (W.infl[p][j] = 0.93 * W.infl[p][j] + 0.07 * spread(o)));
  }
  const next = predictNext(DF, ctx, a, 0, rng);
  W.ctx.push(next); if (W.ctx.length > K + 2) W.ctx.shift();
  // engine ghost: same keys; for autopilot players the real scripted bot decides
  W.eng = step(W.eng, a[0] === null ? botAction(W.eng, 0, rng) : a[0], a[1] === null ? botAction(W.eng, 1, rng) : a[1]);
  W.trail.push(next); if (W.trail.length > 30) W.trail.shift();
  W.t++;
  if (isDiverged(next)) prime();
}

function render() {
  const s = W.ctx[W.ctx.length - 1];
  drawArena(g, 640, 410, s, { title: 'WORLD MODEL (no engine)', color: '#fbbf24', trail: W.trail, ghost: $('#showEngine').checked ? W.eng : null, autopilot: [W.acts[0] === null, W.acts[1] === null] });
  const name = (a) => (a === null ? 'absent → autopilot' : ['no key', '↑', '↓', '←', '→'][a]);
  $('#playInfo').textContent = `t = ${(W.t / 15).toFixed(1)}s · P1: ${name(W.acts[0])} · P2: ${name(W.acts[1])}`;
  // matrix
  const m = $('#matrix');
  const cols = ['car P1', 'car P2', 'ball'];
  let html = '<div class="h"></div>' + cols.map((c) => `<div class="h">${c}</div>`).join('');
  const mx = Math.max(1e-6, ...W.infl.flat());
  W.infl.forEach((row, p) => {
    html += `<div class="h" style="text-align:left">P${p + 1}'s key</div>`;
    row.forEach((v) => { const t = v / mx; html += `<div style="background:rgba(244,114,182,${0.08 + 0.8 * t});color:${t > 0.5 ? '#0b0e14' : '#e6ebf2'}">${(v * 1000).toFixed(1)}</div>`; });
  });
  m.innerHTML = html;
}

let running = false, acc = 0, last = 0;
function loop(ts) {
  if (!running) return;
  const dt = Math.min(100, ts - (last || ts)); last = ts; acc += (dt / 1000) * 15;
  let n = 0; while (acc >= 1 && n < 5) { tick(); acc -= 1; n++; }
  render(); requestAnimationFrame(loop);
}
$('#btnPlay').onclick = () => { running = !running; $('#btnPlay').textContent = running ? '❚❚ Pause' : '▶ Start'; if (running) { last = 0; cv.focus(); requestAnimationFrame(loop); } };
$('#btnKick').onclick = () => { prime(); render(); };
for (const p of ['p1', 'p2']) $$(`#${p}Mode button`).forEach((b) => (b.onclick = () => { mode[p] = b.dataset.m; $$(`#${p}Mode button`).forEach((x) => x.classList.toggle('on', x === b)); }));
window.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (k in K1) { if (!held.p1.includes(k)) held.p1.push(k); if (mode.p1 !== 'keys') $('#p1Mode button[data-m=keys]').click(); e.preventDefault(); }
  if (k in K2) { if (!held.p2.includes(k)) held.p2.push(k); if (mode.p2 !== 'keys') $('#p2Mode button[data-m=keys]').click(); e.preventDefault(); }
});
window.addEventListener('keyup', (e) => { const k = e.key.toLowerCase(); for (const h of [held.p1, held.p2]) { const i = h.indexOf(k); if (i >= 0) h.splice(i, 1); } });
render();

// ------------------------------------------------------------ live evaluation: with vs without P2's keys
setTimeout(() => {
  const eps = genEpisodes(24, 140, 31337);
  const H = 90, errWith = new Float64Array(H), errWithout = new Float64Array(H), carWith = new Float64Array(H), carWithout = new Float64Array(H);
  for (const ep of eps) {
    const acts = ep.A.slice(K - 1, K - 1 + H);
    const rw = rollout(DF, ep.S.slice(0, K), acts, 0, 3);
    const ro = rollout(DF, ep.S.slice(0, K), acts.map(([a]) => [a, null]), 0, 3);
    for (let h = 0; h < H; h++) {
      const gt = ep.S[K + h];
      errWith[h] += Math.hypot(rw[h][8] - gt[8], rw[h][9] - gt[9]) / eps.length;
      errWithout[h] += Math.hypot(ro[h][8] - gt[8], ro[h][9] - gt[9]) / eps.length;
      carWith[h] += Math.hypot(rw[h][4] - gt[4], rw[h][5] - gt[5]) / eps.length;
      carWithout[h] += Math.hypot(ro[h][4] - gt[4], ro[h][5] - gt[5]) / eps.length;
    }
  }
  const pts = (a) => Array.from(a, (v, h) => [(h + 1) / 15, v]);
  lineChart($('#chAttr'), {
    title: 'mean position error vs. ground truth',
    series: [
      { name: 'P2 car · keys given', color: '#fb923c', points: pts(carWith) },
      { name: 'P2 car · keys hidden', color: '#fb923c', dash: '5 4', points: pts(carWithout) },
      { name: 'ball · keys given', color: '#e6ebf2', points: pts(errWith) },
      { name: 'ball · keys hidden', color: '#e6ebf2', dash: '5 4', points: pts(errWithout) },
    ],
    xlabel: 'rollout time (s)', ylabel: 'distance', ymin: 0, height: 260, width: 560,
  });
  const at = 44;
  $('#attrNote').innerHTML = `24 held-out bot matches, same model, same P1 keys. After 3 s the P2 car is off by <b>${carWith[at].toFixed(2)}</b> with its keys vs <b>${carWithout[at].toFixed(2)}</b> without
    (${(carWithout[at] / carWith[at]).toFixed(1)}×), and the ball by ${errWith[at].toFixed(2)} vs ${errWithout[at].toFixed(2)}. Without the keys the model can still play P2 plausibly, but it can't know what the bot <i>actually</i> did.`;
}, 50);
