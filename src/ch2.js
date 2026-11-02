import { mountNav, $, $$ } from './nav.js';
import { textDistribution, makeFlowNet, FlowTrainer, sample, velocityAt, gaussian } from './lib/flow2d.js';
import { MLP } from './lib/nn.js';
import { lineChart } from './lib/charts.js';
mountNav('ch2-flow');

const DATA = textDistribution('MIRA', 4000, 1);
const XR = 3.0, YR = 1.9;
function mapper(cv) {
  const W = cv.width, H = cv.height, s = Math.min(W / (2 * XR), H / (2 * YR));
  return { X: (x) => W / 2 + x * s, Y: (y) => H / 2 - y * s, s };
}
function clear(g, cv) { g.fillStyle = '#0a0d13'; g.fillRect(0, 0, cv.width, cv.height); }
function dots(g, M, pts, color, r = 1.6, alpha = 1, max = Infinity) {
  g.fillStyle = color; g.globalAlpha = alpha;
  const n = Math.min(pts.length / 2, max);
  for (let i = 0; i < n; i++) { const x = M.X(pts[2 * i]), y = M.Y(pts[2 * i + 1]); g.fillRect(x - r / 2, y - r / 2, r, r); }
  g.globalAlpha = 1;
}
const LAT = '#a78bfa', NOISE = '#fb923c', WM = '#fbbf24', GOOD = '#4ade80';

// ---------------------------------------------------------------- 1. interpolation
{
  const cv = $('#cvInterp'), g = cv.getContext('2d'), M = mapper(cv);
  const n = 220, z0 = gaussian(n, 3), z1 = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) { const j = Math.floor(Math.random() * 4000); z1[2 * i] = DATA[2 * j]; z1[2 * i + 1] = DATA[2 * j + 1]; }
  let tau = 0, anim = null;
  const draw = () => {
    clear(g, cv); dots(g, M, DATA, LAT, 1.2, 0.18);
    if ($('#showLines').checked) {
      g.strokeStyle = 'rgba(147,160,180,0.16)'; g.lineWidth = 1;
      for (let i = 0; i < n; i++) { g.beginPath(); g.moveTo(M.X(z0[2 * i]), M.Y(z0[2 * i + 1])); g.lineTo(M.X(z1[2 * i]), M.Y(z1[2 * i + 1])); g.stroke(); }
    }
    for (let i = 0; i < n; i++) {
      const x = tau * z1[2 * i] + (1 - tau) * z0[2 * i], y = tau * z1[2 * i + 1] + (1 - tau) * z0[2 * i + 1];
      const c = tau; g.fillStyle = `rgb(${251 + (167 - 251) * c | 0},${146 + (139 - 146) * c | 0},${60 + (250 - 60) * c | 0})`;
      g.beginPath(); g.arc(M.X(x), M.Y(y), 3, 0, 7); g.fill();
    }
    // one highlighted pair with its constant target velocity
    const i = 7, x = tau * z1[2 * i] + (1 - tau) * z0[2 * i], y = tau * z1[2 * i + 1] + (1 - tau) * z0[2 * i + 1];
    const vx = z1[2 * i] - z0[2 * i], vy = z1[2 * i + 1] - z0[2 * i + 1];
    arrow(g, M.X(x), M.Y(y), M.X(x + vx * 0.35), M.Y(y + vy * 0.35), WM, 2.5);
    g.fillStyle = WM; g.font = '600 12px Inter, sans-serif'; g.fillText('target velocity z₁ − z₀', M.X(x + vx * 0.35) + 6, M.Y(y + vy * 0.35));
    $('#tauCap').textContent = `τ = ${tau.toFixed(2)}  ${tau < 0.05 ? '(pure noise)' : tau > 0.95 ? '(data)' : ''}`;
  };
  $('#tau').oninput = (e) => { tau = +e.target.value; draw(); };
  $('#showLines').onchange = draw;
  $('#tauPlay').onclick = () => {
    if (anim) { cancelAnimationFrame(anim); anim = null; $('#tauPlay').textContent = '▶ animate'; return; }
    $('#tauPlay').textContent = '❚❚ stop'; let dir = 1;
    const tick = () => { tau += dir * 0.008; if (tau >= 1) { tau = 1; dir = -1; } if (tau <= 0) { tau = 0; dir = 1; } $('#tau').value = tau; draw(); anim = requestAnimationFrame(tick); };
    tick();
  };
  draw();
}
function arrow(g, x1, y1, x2, y2, color, w = 1.5) {
  g.strokeStyle = color; g.fillStyle = color; g.lineWidth = w;
  g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
  const a = Math.atan2(y2 - y1, x2 - x1), h = 4 + w * 1.5;
  g.beginPath(); g.moveTo(x2, y2); g.lineTo(x2 - h * Math.cos(a - 0.45), y2 - h * Math.sin(a - 0.45)); g.lineTo(x2 - h * Math.cos(a + 0.45), y2 - h * Math.sin(a + 0.45)); g.fill();
}

// ---------------------------------------------------------------- 2. live training
let net = makeFlowNet(2), trainer = new FlowTrainer(net, DATA), running = false, raf = null, nSteps = 8, fieldTau = 0.5;
const lossPts = [];
const z0s = gaussian(700, 21), z0p = gaussian(120, 33);
const cvS = $('#cvSamples'), gS = cvS.getContext('2d'), MS = mapper(cvS);
const cvP = $('#cvPaths'), gP = cvP.getContext('2d'), MP = mapper(cvP);
const cvF = $('#cvField'), gF = cvF.getContext('2d'), MF = mapper(cvF);

function renderMain() {
  clear(gS, cvS); dots(gS, MS, DATA, LAT, 1.2, 0.12);
  dots(gS, MS, sample(net, z0s, nSteps, false), WM, 2, 0.85);
  clear(gP, cvP); dots(gP, MP, DATA, LAT, 1.2, 0.12);
  const paths = sample(net, z0p, Math.max(nSteps, 1), false, true);
  for (let i = 0; i < z0p.length / 2; i++) {
    gP.strokeStyle = 'rgba(251,191,36,0.45)'; gP.lineWidth = 1; gP.beginPath();
    paths.forEach((p, k) => { const x = MP.X(p[2 * i]), y = MP.Y(p[2 * i + 1]); k ? gP.lineTo(x, y) : gP.moveTo(x, y); });
    gP.stroke();
    const p0 = paths[0], pe = paths[paths.length - 1];
    gP.fillStyle = NOISE; gP.fillRect(MP.X(p0[2 * i]) - 1.5, MP.Y(p0[2 * i + 1]) - 1.5, 3, 3);
    gP.fillStyle = WM; gP.fillRect(MP.X(pe[2 * i]) - 1.5, MP.Y(pe[2 * i + 1]) - 1.5, 3, 3);
  }
  // field
  clear(gF, cvF); dots(gF, MF, DATA, LAT, 1.2, 0.12);
  const grid = []; for (let y = -YR + 0.15; y < YR; y += 0.25) for (let x = -XR + 0.15; x < XR; x += 0.25) grid.push(x, y);
  const G = Float32Array.from(grid), V = velocityAt(net, G, fieldTau);
  for (let i = 0; i < G.length / 2; i++) {
    const vx = V[2 * i], vy = V[2 * i + 1], m = Math.hypot(vx, vy), sc = Math.min(0.11, 0.06 * m) / (m || 1);
    const c = Math.min(1, m / 4);
    arrow(gF, MF.X(G[2 * i]), MF.Y(G[2 * i + 1]), MF.X(G[2 * i] + vx * sc), MF.Y(G[2 * i + 1] + vy * sc), `rgba(${251},${146 + 60 * c | 0},${60},${0.35 + 0.6 * c})`, 1.1);
  }
  $('#fieldCap').textContent = `velocity field at τ=${fieldTau.toFixed(2)}`;
}
function renderLoss() {
  if (lossPts.length < 2) return;
  lineChart($('#lossChart'), { title: 'flow-matching loss', series: [{ name: 'loss (EMA)', color: WM, points: lossPts }], xlabel: 'training step', height: 170, width: 520, legend: false, ymin: Math.min(...lossPts.map((p) => p[1]), 1.2) * 0.95 });
}
let lastRender = 0;
function loop() {
  const t0 = performance.now();
  while (performance.now() - t0 < 40) {
    const st = trainer.steps;
    trainer.fmStep(128, st < 1200 ? 2e-3 : 7e-4);
    if (trainer.steps % 10 === 0) lossPts.push([trainer.steps, trainer.loss]);
  }
  $('#trainInfo').textContent = `step ${trainer.steps} · loss ${trainer.loss.toFixed(3)}`;
  if (performance.now() - lastRender > 450) { renderMain(); renderLoss(); lastRender = performance.now(); }
  if (trainer.steps >= 300 && !student) { $('#btnPSD').disabled = false; $('#btnPSD').textContent = '▶ Distil'; }
  if (running) raf = requestAnimationFrame(loop); else { renderMain(); renderLoss(); }
}
$('#btnTrain').onclick = () => {
  running = !running; $('#btnTrain').textContent = running ? '❚❚ Pause' : '▶ Train';
  if (running) loop(); else cancelAnimationFrame(raf);
};
$('#btnReset').onclick = () => {
  running = false; cancelAnimationFrame(raf); $('#btnTrain').textContent = '▶ Train';
  net = makeFlowNet(Math.floor(Math.random() * 1000)); trainer = new FlowTrainer(net, DATA); lossPts.length = 0;
  $('#trainInfo').textContent = 'step 0 · loss –'; renderMain(); $('#lossChart').innerHTML = '';
};
$$('#stepSel button').forEach((b) => (b.onclick = () => { nSteps = +b.dataset.n; $('#nLbl').textContent = nSteps; $$('#stepSel button').forEach((x) => x.classList.toggle('on', x === b)); renderMain(); }));
$('#fieldTau').oninput = (e) => { fieldTau = +e.target.value; renderMain(); };
renderMain();

// ---------------------------------------------------------------- 3. distillation
const cmp = $$('.cmpc').map((c) => ({ c, g: c.getContext('2d'), M: mapper(c) }));
let student = null, psdTrainer = null, psdRun = false;
const z0c = gaussian(600, 55);
const DSUB = DATA.filter((_, i) => (i >> 1) % 4 === 0);
function quality(z) { // symmetric Chamfer distance between samples and data (punishes collapse)
  const m = Math.min(300, z.length / 2), nd = DSUB.length / 2;
  let a = 0; for (let i = 0; i < m; i++) { let b = 1e9; for (let j = 0; j < nd; j++) { const d = (z[2 * i] - DSUB[2 * j]) ** 2 + (z[2 * i + 1] - DSUB[2 * j + 1]) ** 2; if (d < b) b = d; } a += Math.sqrt(b); }
  let c = 0, cnt = 0; for (let j = 0; j < nd; j += 3) { let b = 1e9; for (let i = 0; i < m; i++) { const d = (z[2 * i] - DSUB[2 * j]) ** 2 + (z[2 * i + 1] - DSUB[2 * j + 1]) ** 2; if (d < b) b = d; } c += Math.sqrt(b); cnt++; }
  return a / m + c / cnt;
}
function renderCmp() {
  const sets = [[net, 1, false], [net, 2, false], [student, 1, true], [student, 2, true]];
  sets.forEach(([m, n, d], i) => {
    const { c, g, M } = cmp[i]; clear(g, c); dots(g, M, DATA, LAT, 1, 0.12);
    if (m) dots(g, M, sample(m, z0c, n, d), i >= 2 ? GOOD : WM, 1.8, 0.85);
  });
}
function renderQual() {
  const Ns = [1, 2, 4, 8, 16];
  const series = [{ name: 'base (flow matching)', color: WM, dots: true, points: Ns.map((n) => [n, quality(sample(net, z0c, n, false))]) }];
  if (student) series.push({ name: 'distilled (PSD)', color: GOOD, dots: true, points: Ns.map((n) => [n, quality(sample(student, z0c, n, true))]) });
  lineChart($('#qualChart'), { title: 'sample error vs. number of steps', series, xlabel: 'sampling steps (log)', ylabel: 'dist. to data ↓', logx: true, xmin: 1, xmax: 16, height: 200, width: 560 });
}
let lastCmp = 0, lastQual = 0;
function psdLoop() {
  const t0 = performance.now();
  while (performance.now() - t0 < 40) psdTrainer.psdStep(128, 1e-3, 0.5);
  $('#psdInfo').textContent = `distillation step ${psdTrainer.steps} · loss ${psdTrainer.loss.toFixed(3)}`;
  if (performance.now() - lastCmp > 400) { renderCmp(); lastCmp = performance.now(); }
  if (performance.now() - lastQual > 2500) { renderQual(); lastQual = performance.now(); }
  if (psdRun) requestAnimationFrame(psdLoop); else { renderCmp(); renderQual(); }
}
$('#btnPSD').onclick = () => {
  if (!student) {
    // warm-start the student from the trained flow-matching model (same architecture, Δ input was always 0)
    student = new MLP(net.sizes); student.W = net.W.map((w) => Float32Array.from(w)); student.b = net.b.map((w) => Float32Array.from(w)); student.initAdam();
    psdTrainer = new FlowTrainer(student, DATA, 77);
    if (running) $('#btnTrain').click();
  }
  psdRun = !psdRun; $('#btnPSD').textContent = psdRun ? '❚❚ Pause distillation' : '▶ Continue distillation';
  if (psdRun) psdLoop(); else renderQual();
};
renderCmp();
