import { mountNav, $, $$ } from './nav.js';
import { kickoff, step, botAction, ARENA } from './lib/toyworld.js';
import { predictNext, K } from './lib/worldmodel.js';
import { unpackMLP, mulberry32, randn } from './lib/nn.js';
import { drawArena, isDiverged } from './lib/toydraw.js';
import { lineChart } from './lib/charts.js';
import { loadImage } from './lib/dino.js';
mountNav('ch3-forcing');

// ------------------------------------------------------------ 1. noise-per-frame illustration
const frameFiles = Array.from({ length: 8 }, (_, i) => `/frames/seq/s_${String(20 + i * 4).padStart(3, '0')}.jpg`);
const frameImgs = await Promise.all(frameFiles.map(loadImage));
const noiseField = (() => { const r = mulberry32(5), n = new Float32Array(256 * 144 * 3); for (let i = 0; i < n.length; i++) n[i] = randn(r); return n; })();
function makeRow(el) {
  el.innerHTML = '';
  return frameImgs.map((im, i) => {
    const d = document.createElement('div'); d.className = 'f';
    const c = document.createElement('canvas'); c.width = 256; c.height = 144;
    const t = document.createElement('div'); t.className = 'tau';
    const b = document.createElement('div'); b.className = 'bar'; b.innerHTML = '<div></div>';
    d.append(c, t, b); el.appendChild(d);
    return { c, t, b, im, i };
  });
}
const rowTF = makeRow($('#framesTF')), rowDF = makeRow($('#framesDF'));
function paint(cell, tau, target) {
  const g = cell.c.getContext('2d');
  g.drawImage(cell.im, 0, 0, 256, 144, 0, 0, 256, 144); // top-left player view
  const id = g.getImageData(0, 0, 256, 144), d = id.data;
  for (let p = 0, q = 0; p < d.length; p += 4, q += 3) for (let k = 0; k < 3; k++) {
    const x = d[p + k] / 127.5 - 1; const z = tau * x + (1 - tau) * noiseField[q + k] * 0.9;
    d[p + k] = Math.max(0, Math.min(255, (z + 1) * 127.5));
  }
  g.putImageData(id, 0, 0);
  if (target) { g.strokeStyle = '#fb923c'; g.lineWidth = 5; g.strokeRect(2, 2, 252, 140); }
  cell.t.innerHTML = `τ=${tau.toFixed(2)}${target ? ' <span class="c-noise">●</span>' : ''}`;
  cell.b.firstChild.style.width = `${(1 - tau) * 100}%`;
}
function resample() {
  rowTF.forEach((c, i) => paint(c, i < 7 ? 1 : Math.random(), i === 7));
  rowDF.forEach((c) => paint(c, Math.random(), true));
}
$('#resample').onclick = resample; resample();

// ------------------------------------------------------------ 2. three worlds
const load = async (m) => unpackMLP(await (await fetch(`/models/toy/${m}.json`)).json());
let TF = await load('teacher'), DF = await load('diffusion');

const cvs = { true: $('#cvTrue'), tf: $('#cvTF'), df: $('#cvDF') };
const gs = Object.fromEntries(Object.entries(cvs).map(([k, c]) => [k, c.getContext('2d')]));
let rng = mulberry32(7), W;
const pressed = []; // stack of held movement keys
const KEYMAP = { w: 1, s: 2, a: 3, d: 4 };
window.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase(); if (!(k in KEYMAP)) return;
  if (!pressed.includes(k)) pressed.push(k); $('#autoP1').checked = false; e.preventDefault();
  $$('.kbd[data-k]').forEach((el) => el.classList.toggle('down', pressed.includes(el.dataset.k)));
});
window.addEventListener('keyup', (e) => { const k = e.key.toLowerCase(); const i = pressed.indexOf(k); if (i >= 0) pressed.splice(i, 1); $$('.kbd[data-k]').forEach((el) => el.classList.toggle('down', pressed.includes(el.dataset.k))); });

function prime() {
  // Like MIRA's prefill: every world starts from the same K real frames of the engine.
  let s = kickoff(rng); const hist = [s];
  for (let i = 0; i < K - 1; i++) { s = step(s, botAction(s, 0, rng), botAction(s, 1, rng)); hist.push(s); }
  W = {
    t: 0, engine: hist.slice(), tf: hist.slice(), df: hist.slice(),
    tfDiv: null, dfDiv: null, trail: { engine: [], tf: [], df: [] },
    errTF: [], errDF: [], spdE: [], spdTF: [], spdDF: [],
  };
}
prime();

function tick() {
  const s = W.engine[W.engine.length - 1];
  const a1 = $('#autoP1').checked ? botAction(s, 0, rng) : pressed.length ? KEYMAP[pressed[pressed.length - 1]] : 0;
  const a2 = botAction(s, 1, rng);
  const sn = step(s, a1, a2);
  push(W.engine, sn); W.t++;
  const noise = +$('#ctxNoise').value;
  if (!W.tfDiv) { const p = predictNext(TF, W.tf.slice(-K), [a1, a2], 0, rng); push(W.tf, p); if (isDiverged(p)) W.tfDiv = W.t; }
  if (!W.dfDiv) { const p = predictNext(DF, W.df.slice(-K), [a1, a2], noise, rng); push(W.df, p); if (isDiverged(p)) W.dfDiv = W.t; }
  const e = W.engine.at(-1), tf = W.tf.at(-1), df = W.df.at(-1);
  const sec = W.t / 15;
  const err = (p) => Math.min(5, Math.hypot(p[8] - e[8], p[9] - e[9]));
  if (!W.tfDiv) W.errTF.push([sec, err(tf)]); if (!W.dfDiv) W.errDF.push([sec, err(df)]);
  const spd = (p) => Math.max(1e-3, Math.min(1e4, Math.hypot(p[10], p[11]) * 15));
  W.spdE.push([sec, spd(e)]); W.spdTF.push([sec, W.tfDiv ? 1e4 : spd(tf)]); W.spdDF.push([sec, W.dfDiv ? 1e4 : spd(df)]);
  for (const k of ['engine', 'tf', 'df']) { W.trail[k].push(W[k].at(-1)); if (W.trail[k].length > 40) W.trail[k].shift(); }
}
function push(arr, s) { arr.push(s); if (arr.length > K + 2) arr.shift(); }

function render() {
  const e = W.engine.at(-1);
  drawArena(gs.true, 420, 270, e, { title: 'ENGINE', color: '#e6ebf2', trail: W.trail.engine });
  drawArena(gs.tf, 420, 270, W.tf.at(-1), { title: 'TEACHER-FORCED MODEL', color: '#38bdf8', trail: W.trail.tf, diverged: !!W.tfDiv, ghost: e });
  drawArena(gs.df, 420, 270, W.df.at(-1), { title: 'DIFFUSION-FORCED MODEL', color: '#fb923c', trail: W.trail.df, diverged: !!W.dfDiv, ghost: e });
  const sec = (W.t / 15).toFixed(1);
  $('#tTrue').textContent = `t = ${sec}s`;
  $('#tTF').textContent = W.tfDiv ? `diverged at ${(W.tfDiv / 15).toFixed(1)}s` : `t = ${sec}s`;
  $('#tDF').textContent = W.dfDiv ? `diverged at ${(W.dfDiv / 15).toFixed(1)}s` : `t = ${sec}s`;
}
let lastChart = 0;
function charts() {
  const win = (a) => a.filter((p) => p[0] > W.t / 15 - 60);
  lineChart($('#chErr'), { title: 'ball-position error vs. engine', series: [{ name: 'teacher', color: '#38bdf8', points: win(W.errTF) }, { name: 'diffusion', color: '#fb923c', points: win(W.errDF) }], xlabel: 'seconds (game time)', ymin: 0, ymax: 2.2, height: 210, width: 520 });
  lineChart($('#chSpd'), { title: 'ball speed (log)', series: [{ name: 'engine', color: '#e6ebf2', points: win(W.spdE) }, { name: 'teacher', color: '#38bdf8', points: win(W.spdTF) }, { name: 'diffusion', color: '#fb923c', points: win(W.spdDF) }], xlabel: 'seconds (game time)', logy: true, ymin: 0.01, ymax: 1e4, height: 210, width: 520 });
}

let running = false, acc = 0, last = 0;
function loop(ts) {
  if (!running) return;
  const dt = Math.min(100, ts - (last || ts)); last = ts;
  acc += (dt / 1000) * 15 * +$('#speed').value;
  let n = 0; while (acc >= 1 && n < 40) { tick(); acc -= 1; n++; }
  render();
  if (ts - lastChart > 300) { charts(); lastChart = ts; }
  requestAnimationFrame(loop);
}
$('#btnRun').onclick = () => { running = !running; $('#btnRun').textContent = running ? '❚❚ Pause' : '▶ Run'; if (running) { last = 0; requestAnimationFrame(loop); } };
$('#btnReset').onclick = () => { prime(); render(); charts(); };
$('#ctxNoise').oninput = (e) => ($('#ctxNoiseV').textContent = (+e.target.value).toFixed(2));
render(); charts();

// optional in-browser retraining
$('#btnRetrain').onclick = () => {
  $('#btnRetrain').disabled = true;
  const prog = { teacher: 0, diffusion: 0 }, done = {};
  for (const mode of ['teacher', 'diffusion']) {
    const w = new Worker(new URL('./toytrain.worker.js', import.meta.url), { type: 'module' });
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') { prog[mode] = m.step / m.steps; $('#retrainInfo').textContent = `teacher ${(prog.teacher * 100).toFixed(0)}% · diffusion ${(prog.diffusion * 100).toFixed(0)}% (loss ${m.loss.toFixed(4)})`; }
      if (m.type === 'done') {
        done[mode] = unpackMLP(m.model); w.terminate();
        if (done.teacher && done.diffusion) { TF = done.teacher; DF = done.diffusion; prime(); render(); charts(); $('#retrainInfo').textContent = 'now using the models you just trained (3,000 steps each, 300 matches)'; $('#btnRetrain').disabled = false; }
      }
    };
    w.postMessage({ mode, steps: 3000, episodes: 300, seed: Math.floor(Math.random() * 1000) });
  }
};

// ------------------------------------------------------------ 3. Fig. 10 chart
const T = [2, 4, 6, 8, 10, 12];
const mk = (arr) => arr.map((v, i) => [T[i], v]);
lineChart($('#chFig10'), {
  title: 'gFID over a 12 s rollout',
  series: [
    { name: 'distilled codec, noise 0', color: '#f87171', dots: true, points: mk([13.6, 17.2, 23.1, 33.9, 53.7, 81.7]) },
    { name: 'distilled, noise 0.2', color: '#fbbf24', dots: true, points: mk([13.3, 15.5, 15.8, 15.9, 16.7, 16.9]) },
    { name: 'DINOv3 (MIRA), noise 0', color: '#4ade80', dots: true, points: mk([8.7, 10.7, 10.8, 11.6, 12.5, 12.7]) },
    { name: 'noise 0.2', color: '#2dd4bf', dash: '5 4', dots: true, points: mk([8.5, 10.8, 11.3, 11.8, 13.0, 12.6]) },
  ],
  xlabel: 'rollout time (s)', ylabel: 'gFID ↓', ymin: 0, ymax: 85, height: 280, width: 560,
});
