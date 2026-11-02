import { mountNav, $, $$ } from './nav.js';
import { loadImage } from './lib/dino.js';
mountNav('ch6-realtime');

const NFR = 180;
const frames = await Promise.all(Array.from({ length: NFR }, (_, i) => loadImage(`/frames/seq/s_${String(i + 1).padStart(3, '0')}.jpg`)));

// ---------------------------------------------------------------- 1. rolling window
{
  const cv = $('#cvWin'), g = cv.getContext('2d');
  const WIN = 20; let slots = [], next = 0, anim = 0, auto = null;
  const reset = () => { slots = []; for (let k = 0; k < WIN; k++) slots.push({ kind: 'real', f: 2 * k }); next = 2 * WIN; anim = 0; draw(); };
  const noise = (() => { const c = document.createElement('canvas'); c.width = 64; c.height = 36; const x = c.getContext('2d'), d = x.createImageData(64, 36); for (let i = 0; i < d.data.length; i += 4) { const v = Math.random() * 255; d.data[i] = d.data[i + 1] = d.data[i + 2] = v; d.data[i + 3] = 255; } x.putImageData(d, 0, 0); return c; })();
  function draw() {
    g.fillStyle = '#151b26'; g.fillRect(0, 0, cv.width, cv.height);
    const sw = 42, sh = 24 * 2 + 4, x0 = 60, y0 = 56, gap = 2.5;
    g.font = '600 11px ui-monospace,monospace';
    g.fillStyle = '#93a0b4'; g.fillText('dropped', 2, y0 + 30);
    slots.forEach((s, k) => {
      const x = x0 + k * (sw + gap);
      // each latent = 2 frames: show both, stacked
      g.drawImage(frames[s.f % NFR], 0, 0, 256, 144, x, y0, sw, 24);
      g.drawImage(frames[(s.f + 1) % NFR], 0, 0, 256, 144, x, y0 + 26, sw, 24);
      g.strokeStyle = s.kind === 'real' ? '#4ade80' : '#fbbf24'; g.lineWidth = 2; g.strokeRect(x - 1, y0 - 1, sw + 2, sh + 1);
      if (k >= WIN - 3) { g.fillStyle = '#38bdf8'; g.fillRect(x, y0 + sh + 6, sw, 3); }
    });
    g.fillStyle = '#38bdf8'; g.fillText('decoder reads last 3', x0 + (WIN - 3) * (sw + gap), y0 + sh + 22);
    // incoming slot
    const xi = x0 + WIN * (sw + gap) + 18;
    g.globalAlpha = 1 - anim; g.drawImage(noise, xi, y0, sw * 1.3, 24); g.drawImage(noise, xi, y0 + 26, sw * 1.3, 24);
    g.globalAlpha = anim; g.drawImage(frames[next % NFR], 0, 0, 256, 144, xi, y0, sw * 1.3, 24); g.drawImage(frames[(next + 1) % NFR], 0, 0, 256, 144, xi, y0 + 26, sw * 1.3, 24);
    g.globalAlpha = 1; g.strokeStyle = '#fb923c'; g.setLineDash([4, 3]); g.strokeRect(xi - 1, y0 - 1, sw * 1.3 + 2, sh + 1); g.setLineDash([]);
    g.fillStyle = '#fb923c'; g.fillText('new latent', xi, y0 - 20); g.fillText(anim < 1 ? `denoise ${(anim * 100) | 0}%` : 'append', xi, y0 - 7);
    g.fillStyle = '#e6ebf2'; g.font = '700 12px Inter, sans-serif';
    g.fillText('context window (20 latents = 2 s, each latent decodes to 2 frames)', x0, 18);
    g.font = '11px Inter, sans-serif'; g.fillStyle = '#4ade80'; g.fillText('■ real (prefill)', x0, 34); g.fillStyle = '#fbbf24'; g.fillText('■ generated', x0 + 110, 34);
    const nGen = slots.filter((s) => s.kind === 'gen').length;
    g.fillStyle = '#93a0b4'; g.fillText(nGen === 0 ? 'phase: PREFILL done — every step from now on is generated' : nGen < WIN ? `phase: DECODE — ${nGen}/20 slots generated` : 'phase: DECODE — the window is now 100% the model’s own output', x0 + 230, 34);
    g.fillStyle = '#66728a'; g.fillText('oldest  ←', x0, y0 + sh + 40); g.fillText('→  newest', x0 + (WIN - 1) * (sw + gap) - 20, y0 + sh + 40);
  }
  function stepOnce() {
    if (anim < 1) { const t0 = performance.now(); const a = () => { anim = Math.min(1, (performance.now() - t0) / 350); draw(); if (anim < 1) requestAnimationFrame(a); else commit(); }; a(); }
  }
  function commit() { slots.push({ kind: 'gen', f: next }); slots.shift(); next += 2; anim = 0; setTimeout(draw, 120); }
  $('#winStep').onclick = stepOnce;
  $('#winAuto').onclick = () => { if (auto) { clearInterval(auto); auto = null; $('#winAuto').textContent = 'auto'; } else { auto = setInterval(stepOnce, 600); $('#winAuto').textContent = 'stop'; } };
  $('#winReset').onclick = reset;
  reset();
}

// ---------------------------------------------------------------- 2. serving simulation
const P = {};
const sliders = ['nfe', 'msNfe', 'msDec', 'qcap', 'jit'];
function readP() { sliders.forEach((k) => { P[k] = +$('#' + k).value; $('#' + k + 'V').textContent = P[k] + (k === 'nfe' || k === 'qcap' ? '' : ' ms'); }); P.kv = $('#kv').checked; showStep(); }
function stepCost() { return P.nfe * P.msNfe * (P.kv ? 1 : 20) + P.msDec; }
function showStep() {
  const c = stepCost(); $('#stepMs').textContent = `${c} ms`;
  const pf = c / 2; $('#perFrame').textContent = `${pf.toFixed(0)} ms`; $('#perFrame').style.color = pf <= 50 ? 'var(--good)' : 'var(--bad)';
}
sliders.forEach((k) => ($('#' + k).oninput = readP)); $('#kv').onchange = readP;
const PRESETS = { mira: { nfe: 2, msNfe: 25, msDec: 20, qcap: 4, jit: 8, kv: true }, nodistill: { nfe: 10, msNfe: 25, msDec: 20, qcap: 4, jit: 8, kv: true }, nocache: { nfe: 2, msNfe: 25, msDec: 20, qcap: 4, jit: 8, kv: false }, noqueue: { nfe: 2, msNfe: 25, msDec: 20, qcap: 0, jit: 8, kv: true } };
$$('.presets button').forEach((b) => (b.onclick = () => { const p = PRESETS[b.dataset.p]; sliders.forEach((k) => ($('#' + k).value = p[k])); $('#kv').checked = p.kv; readP(); resetSim(); simTarget = 0; $$('.presets button').forEach((x) => x.classList.toggle('on', x === b)); }));
readP();

const screen = $('#cvScreen').getContext('2d');
let sim;
function resetSim() {
  sim = { t: 0, phase: 'wm', phaseEnd: dur('wm'), stepStart: 0, queue: [], nextFrame: 0, shown: 0, lastShown: null, nextTick: 50, ev: [], ticks: [], lat: [], stalls: 0, newShown: [] };
}
function dur(ph) { const j = (Math.random() * 2 - 1) * P.jit; return Math.max(2, (ph === 'wm' ? P.nfe * P.msNfe * (P.kv ? 1 : 20) : P.msDec) + j); }
function advance(to) {
  while (sim.t < to) {
    const tNext = Math.min(to, sim.phaseEnd ?? Infinity, sim.nextTick);
    sim.t = tNext;
    if (sim.phaseEnd !== null && sim.t >= sim.phaseEnd) {
      if (sim.phase === 'wm') { sim.ev.push({ lane: 0, a: sim.phaseStart ?? sim.stepStart, b: sim.t }); sim.phase = 'dec'; sim.phaseStart = sim.t; sim.phaseEnd = sim.t + dur('dec'); }
      else if (sim.phase === 'dec') { sim.ev.push({ lane: 1, a: sim.phaseStart, b: sim.t }); sim.phase = 'push'; sim.phaseEnd = sim.t; }
      if (sim.phase === 'push') {
        if (P.qcap === 0) { // no queue: frames hit the screen as soon as they exist (both at once)
          for (let k = 0; k < 2; k++) { sim.lastShown = { f: sim.nextFrame++, born: sim.stepStart }; }
          sim.newShown.push(sim.t); sim.lat.push(sim.t - sim.stepStart);
          sim.direct = true; startStep();
        } else if (P.qcap - sim.queue.length >= 2) {
          for (let k = 0; k < 2; k++) sim.queue.push({ f: sim.nextFrame++, born: sim.stepStart });
          startStep();
        } else { sim.phaseEnd = null; if (!sim.blockStart) sim.blockStart = sim.t; } // backpressure: wait for space
      }
    }
    if (sim.t >= sim.nextTick) {
      // display tick
      let fresh = false;
      if (P.qcap > 0 && sim.queue.length) { const fr = sim.queue.shift(); sim.lastShown = fr; fresh = true; sim.lat.push(sim.t - fr.born); sim.newShown.push(sim.t); }
      else if (P.qcap === 0 && sim.direct) { fresh = true; sim.direct = false; }
      if (!fresh) sim.stalls++;
      sim.ticks.push({ t: sim.t, fresh });
      sim.qlog = sim.qlog || []; sim.qlog.push({ t: sim.t, n: sim.queue.length });
      sim.nextTick += 50;
      if (sim.phase === 'push' && sim.phaseEnd === null && P.qcap - sim.queue.length >= 2) {
        sim.ev.push({ lane: 2, a: sim.blockStart, b: sim.t }); sim.blockStart = null;
        for (let k = 0; k < 2; k++) sim.queue.push({ f: sim.nextFrame++, born: sim.stepStart });
        startStep();
      }
    }
  }
  const cut = sim.t - 3000;
  sim.ev = sim.ev.filter((e) => e.b > cut); sim.ticks = sim.ticks.filter((e) => e.t > cut); sim.newShown = sim.newShown.filter((t) => t > sim.t - 2000);
  if (sim.qlog) sim.qlog = sim.qlog.filter((e) => e.t > cut); if (sim.lat.length > 40) sim.lat = sim.lat.slice(-40);
}
function startStep() { sim.phase = 'wm'; sim.stepStart = sim.t; sim.phaseStart = sim.t; sim.phaseEnd = sim.t + dur('wm'); }

const gG = $('#cvGantt').getContext('2d');
function drawGantt() {
  const W = 1100, H = 200, L = 120, span = 2000, t1 = sim.t, t0 = t1 - span;
  const X = (t) => L + ((t - t0) / span) * (W - L - 10);
  gG.fillStyle = '#0a0d13'; gG.fillRect(0, 0, W, H);
  const lanes = [['GPU · world model', 20, '#fbbf24'], ['GPU · decoder', 50, '#38bdf8'], ['producer blocked', 80, '#93a0b4'], ['queue', 112, '#a78bfa'], ['display (50 ms)', 152, '#4ade80']];
  gG.font = '600 11px Inter, sans-serif';
  lanes.forEach(([n, y, c]) => { gG.fillStyle = c; gG.fillText(n, 8, y + 14); gG.fillStyle = 'rgba(255,255,255,0.04)'; gG.fillRect(L, y, W - L - 10, 22); });
  for (let t = Math.ceil(t0 / 100) * 100; t < t1; t += 100) { gG.fillStyle = '#263041'; gG.fillRect(X(t), 10, 1, H - 20); }
  sim.ev.forEach((e) => {
    const y = [20, 50, 80][e.lane], c = ['#fbbf24', '#38bdf8', '#5b6b85'][e.lane];
    gG.fillStyle = c; gG.fillRect(Math.max(L, X(e.a)), y + 2, Math.max(1, X(e.b) - Math.max(L, X(e.a)) - 1), 18);
  });
  if (sim.phase === 'wm' || sim.phase === 'dec') { const y = sim.phase === 'wm' ? 20 : 50; gG.fillStyle = sim.phase === 'wm' ? 'rgba(251,191,36,.5)' : 'rgba(56,189,248,.5)'; gG.fillRect(Math.max(L, X(sim.phaseStart)), y + 2, X(sim.t) - Math.max(L, X(sim.phaseStart)), 18); }
  if (sim.blockStart) { gG.fillStyle = 'rgba(147,160,180,.5)'; gG.fillRect(Math.max(L, X(sim.blockStart)), 82, X(sim.t) - Math.max(L, X(sim.blockStart)), 18); }
  (sim.qlog || []).forEach((q, i, a) => { const nx = a[i + 1] ? a[i + 1].t : sim.t; const h = P.qcap ? (q.n / Math.max(1, P.qcap)) * 22 : 0; gG.fillStyle = '#a78bfa'; gG.fillRect(Math.max(L, X(q.t)), 134 - h, Math.max(0, X(nx) - Math.max(L, X(q.t))), h); });
  sim.ticks.forEach((k) => { if (k.t < t0) return; gG.fillStyle = k.fresh ? '#4ade80' : '#f87171'; gG.fillRect(X(k.t) - 3, 154, 6, 18); });
  gG.fillStyle = '#66728a'; gG.fillText('← 2 s of simulated time →', W / 2 - 60, H - 4);
}

let running = false, lastTs = 0, simTarget = 0;
function loop(ts) {
  if (!running) return;
  const dt = Math.min(100, ts - (lastTs || ts)); lastTs = ts;
  simTarget += dt * +$('#rtScale').value;
  advance(simTarget);
  if (sim.lastShown) screen.drawImage(frames[sim.lastShown.f % NFR], 0, 0, 512, 288);
  $('#stFps').textContent = (P.qcap === 0 ? Math.min(20, 2000 / stepCost()) : sim.newShown.length / 2).toFixed(1);
  $('#stFps').style.color = (P.qcap === 0 ? 0 : sim.newShown.length / 2) >= 19 ? 'var(--good)' : 'var(--bad)';
  $('#stStall').textContent = sim.stalls;
  $('#stLat').textContent = sim.lat.length ? `${(sim.lat.reduce((a, b) => a + b, 0) / sim.lat.length).toFixed(0)} ms` : '–';
  drawGantt();
  requestAnimationFrame(loop);
}
$('#rtRun').onclick = () => { running = !running; $('#rtRun').textContent = running ? '❚❚ Pause' : '▶ Run'; if (running) { lastTs = 0; requestAnimationFrame(loop); } };
resetSim(); simTarget = 0; screen.drawImage(frames[0], 0, 0, 512, 288); drawGantt();
