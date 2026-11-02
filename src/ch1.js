import { mountNav, $, $$, viridis } from './nav.js';
import { getDino, loadImage, cropRGBA, SOURCES } from './lib/dino.js';
import { pca, project } from './lib/linalg.js';
import { barChart, svgEl } from './lib/charts.js';
mountNav('ch1-codec');

// ------------------------------------------------------------------ diagram
(function diagram() {
  const svg = $('#codecSvg');
  const box = (x, y, w, h, color, title, sub, fill = 0.1) => {
    svgEl('rect', { x, y, width: w, height: h, rx: 10, fill: color, 'fill-opacity': fill, stroke: color, 'stroke-width': 1.5 }, svg);
    svgEl('text', { x: x + w / 2, y: y + h / 2 - (sub ? 6 : -4), fill: '#e6ebf2', 'font-size': 13, 'font-weight': 700, 'text-anchor': 'middle' }, svg).textContent = title;
    if (sub) sub.split('\n').forEach((s, i) => (svgEl('text', { x: x + w / 2, y: y + h / 2 + 12 + i * 14, fill: '#93a0b4', 'font-size': 11, 'text-anchor': 'middle' }, svg).textContent = s));
  };
  const arrow = (x1, y1, x2, y2, color = '#5b6b85', label) => {
    svgEl('line', { x1, y1, x2: x2 - 6, y2, stroke: color, 'stroke-width': 1.8 }, svg);
    svgEl('path', { d: `M${x2},${y2} l-8,-5 v10 z`, fill: color }, svg);
    if (label) svgEl('text', { x: (x1 + x2) / 2, y: y1 - 6, fill: '#93a0b4', 'font-size': 10.5, 'text-anchor': 'middle' }, svg).textContent = label;
  };
  const ENC = '#2dd4bf', LAT = '#a78bfa', DEC = '#38bdf8', WM = '#fbbf24';
  svgEl('text', { x: 20, y: 22, fill: ENC, 'font-size': 12, 'font-weight': 700, 'letter-spacing': '0.1em' }, svg).textContent = 'ENCODER';
  svgEl('text', { x: 1090, y: 22, 'text-anchor': 'end', fill: DEC, 'font-size': 12, 'font-weight': 700, 'letter-spacing': '0.1em' }, svg).textContent = 'DECODER';
  const B = (x, w, h, c, t, sub, f) => box(x, 150 - h / 2, w, h, c, t, sub, f);
  B(8, 122, 104, '#93a0b4', '2 frames', '288 (px tall) ×\n512 (px wide) ×\n3 (RGB) @ 20 fps', 0.06);
  arrow(130, 150, 150, 150);
  B(150, 152, 118, ENC, 'DINOv3-L/16', 'FROZEN · per frame\n18 (rows) × 32 (cols)\n× 1024 (features)\navg of layers 11–23', 0.12);
  arrow(302, 150, 322, 150);
  B(322, 152, 104, LAT, 'Linear 2×2×2', 'merges 2 (rows) × 2 (cols)\n× 2 (frames);\n1024 → 32 (channels)', 0.12);
  arrow(474, 150, 494, 150);
  B(494, 142, 118, LAT, 'latent z', '9 (rows) × 16 (cols)\n× 32 (channels)\n@ 10 Hz\n= 4,608 numbers', 0.28);
  arrow(636, 150, 656, 150);
  B(656, 140, 90, DEC, 'Upsample', '9 (rows) × 16 (cols)\n→ 18 (rows) × 32 (cols)', 0.1);
  arrow(796, 150, 816, 150);
  B(816, 140, 104, DEC, 'Space-time ViT-XL', 'TRAINED · 0.45B\nfull attn in space\ncausal in time', 0.12);
  arrow(956, 150, 974, 150);
  B(974, 120, 90, DEC, 'Temporal ×2', '1 latent → 2 frames\n50 ms apart', 0.1);
  // world model hook
  svgEl('path', { d: 'M565,91 C565,42 565,42 565,42', stroke: WM, 'stroke-width': 1.5, 'stroke-dasharray': '5 4', fill: 'none' }, svg);
  box(450, 8, 230, 34, WM, 'World model predicts HERE', '', 0.15);
  svgEl('text', { x: 565, y: 232, fill: '#93a0b4', 'font-size': 11.5, 'text-anchor': 'middle' }, svg).textContent = 'deterministic: no noise, no KL';
  svgEl('text', { x: 870, y: 232, fill: '#93a0b4', 'font-size': 11.5, 'text-anchor': 'middle' }, svg).textContent = 'loss = L1 + λp·LPIPS + λd·‖DINO(x) − DINO(x̂)‖²';
  svgEl('text', { x: 870, y: 249, fill: '#66728a', 'font-size': 11, 'text-anchor': 'middle' }, svg).textContent = 'λ’s rebalanced every step by gradient norm · no GAN';
  svgEl('text', { x: 226, y: 232, fill: '#66728a', 'font-size': 11, 'text-anchor': 'middle' }, svg).textContent = 'mean of chosen layers + the deepest one';
  svgEl('text', { x: 20, y: 285, fill: '#66728a', 'font-size': 11 }, svg).textContent = 'Each player view is encoded and decoded independently with the same codec (600M params total incl. the frozen 300M DINOv3-L).';
})();

// ------------------------------------------------------------------ state
const dino = getDino();
const S = {
  src: 'mira', frame: 0, mode: 'pixels', alpha: 0.85,
  imgs: [], feats: null, // feats: array of 4 Float32Array(576*384)
  D: 384, gh: 18, gw: 32,
  query: null, // {v, r, c}
  cache: new Map(),
  latChan: 32, pool: true, latMode: 'pca',
};

const viewer = $('#viewer');
const canv = [], ctxs = [];
for (let v = 0; v < 4; v++) {
  const d = document.createElement('div'); d.className = 'v';
  const c = document.createElement('canvas'); c.width = 512; c.height = 288;
  const l = document.createElement('div'); l.className = 'lbl'; l.textContent = `Player ${v + 1}`;
  d.append(c, l); viewer.appendChild(d); canv.push(c); ctxs.push(c.getContext('2d'));
  c.addEventListener('click', (e) => {
    const r = c.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * 512, y = ((e.clientY - r.top) / r.height) * 288;
    S.query = { v, r: Math.floor(y / 16), c: Math.floor(x / 16) };
    if (S.feats) { if (S.mode !== 'sim') setMode('sim'); else draw(); drawLatent(); }
  });
}

// sources
const srcSel = $('#srcSel');
for (const k in SOURCES) srcSel.add(new Option(SOURCES[k].label, k));
srcSel.onchange = () => { S.src = srcSel.value; S.frame = 0; buildThumbs(); selectFrame(0); };
function buildThumbs() {
  const t = $('#thumbs'); t.innerHTML = '';
  SOURCES[S.src].frames.forEach((f, i) => {
    const im = document.createElement('img'); im.src = f; if (i === S.frame) im.className = 'on';
    im.onclick = () => selectFrame(i); t.appendChild(im);
  });
}

async function selectFrame(i) {
  S.frame = i; $$('#thumbs img').forEach((im, j) => im.classList.toggle('on', j === i));
  const src = SOURCES[S.src], img = await loadImage(src.frames[i]);
  S.imgs = src.views.map((v) => cropRGBA(img, ...v));
  S.feats = S.cache.get(src.frames[i]) || null; S.basis = null; S.pcs = null; S.latBasis = null;
  draw(); drawLatent();
  if (dino.provider && !S.feats) encodeCurrent();
}

$$('#modeSeg button').forEach((b) => (b.onclick = () => setMode(b.dataset.m)));
function setMode(m) {
  S.mode = m; $$('#modeSeg button').forEach((b) => b.classList.toggle('on', b.dataset.m === m));
  if (m === 'sim' && !S.query) S.query = { v: 0, r: 9, c: 16 };
  draw();
}
$('#pcSel').onchange = (e) => { S.pc0 = +e.target.value; S.pcs = null; draw(); };
$('#alpha').oninput = (e) => { S.alpha = +e.target.value; draw(); };

// ------------------------------------------------------------------ model loading / encoding
$('#btnLoad').onclick = async () => {
  $('#btnLoad').disabled = true;
  $('#status').innerHTML = '<span class="spin"></span> downloading weights…';
  dino.onprogress = (p) => ($('#prog').style.width = (p * 100).toFixed(0) + '%');
  try {
    const prov = await dino.init('fp32', new URLSearchParams(location.search).get('ep') || $('#epSel').value);
    $('#prog').style.width = '100%';
    $('#status').textContent = `loaded · backend: ${prov.toUpperCase()}`;
    await encodeCurrent();
    $('#btnLoad').textContent = 'Model loaded';
  } catch (e) { $('#status').textContent = 'failed: ' + e.message; $('#btnLoad').disabled = false; }
};

async function encodeCurrent() {
  const key = SOURCES[S.src].frames[S.frame];
  if (S.cache.has(key)) { S.feats = S.cache.get(key); S.basis = null; S.pcs = null; S.latBasis = null; draw(); drawLatent(); return; }
  const feats = []; let ms = 0;
  for (let v = 0; v < 4; v++) {
    $('#status').innerHTML = `<span class="spin"></span> encoding view ${v + 1}/4 …`;
    const r = await dino.embed(S.imgs[v].rgba, 512, 288);
    feats.push(r.feats); ms += r.ms; S.D = r.D; S.gh = r.gh; S.gw = r.gw;
  }
  S.cache.set(key, feats);
  if (SOURCES[S.src].frames[S.frame] !== key) return; // user moved on
  S.feats = feats; S.basis = null; S.pcs = null; S.latBasis = null;
  $('#status').textContent = `encoded 4 views · ${(ms / 4).toFixed(0)} ms / view on ${dino.provider.toUpperCase()} · 576 patches (18 rows × 32 columns) × ${S.D} features each`;
  if (S.mode === 'pixels') setMode('pca'); else draw();
  drawLatent();
}

// ------------------------------------------------------------------ feature math
function stackFeats(F, N, D) {
  const X = new Float32Array(F.length * N * D);
  F.forEach((f, i) => X.set(f, i * N * D));
  return X;
}
function ensureBasis() {
  if (S.basis) return;
  const N = S.gh * S.gw, D = S.D;
  const X = stackFeats(S.feats, N, D);
  S.basis = pca(X, 4 * N, D); S.X = X;
  S.normed = normalizeRows(centerRows(X, 4 * N, D, S.basis.mean), 4 * N, D);
}
function centerRows(X, N, D, mean) { const Y = new Float32Array(X.length); for (let i = 0; i < N; i++) for (let d = 0; d < D; d++) Y[i * D + d] = X[i * D + d] - mean[d]; return Y; }
function normalizeRows(X, N, D) { for (let i = 0; i < N; i++) { let s = 0; for (let d = 0; d < D; d++) s += X[i * D + d] ** 2; s = Math.sqrt(s) || 1; for (let d = 0; d < D; d++) X[i * D + d] /= s; } return X; }
// Normalise the query's own view by its best non-self match; the other three views share one scale
// (best match in any other view) so you can compare how strongly each of them responds.
function viewMax(sim, N, qv, qi) {
  let own = 1e-6, other = 1e-6;
  for (let i = 0; i < sim.length; i++) { if (i === qi) continue; if (Math.floor(i / N) === qv) own = Math.max(own, sim[i]); else other = Math.max(other, sim[i]); }
  return [0, 1, 2, 3].map((v) => (v === qv ? own : other));
}
function pctl(arr, p) { const a = Float32Array.from(arr).sort(); return a[Math.floor(p * (a.length - 1))]; }
function toRGB(P, N, k = 3) {
  const lo = [], hi = [];
  for (let c = 0; c < 3; c++) { const col = new Float32Array(N); for (let i = 0; i < N; i++) col[i] = P[i * k + c]; lo.push(pctl(col, 0.02)); hi.push(pctl(col, 0.98)); }
  return (i) => [0, 1, 2].map((c) => Math.max(0, Math.min(255, ((P[i * k + c] - lo[c]) / (hi[c] - lo[c] || 1)) * 255)));
}
function simTo(normed, N, D, qi) {
  const out = new Float32Array(N); const qo = qi * D;
  for (let i = 0; i < N; i++) { let s = 0; const o = i * D; for (let d = 0; d < D; d++) s += normed[o + d] * normed[qo + d]; out[i] = s; }
  return out;
}

// ------------------------------------------------------------------ drawing (full-res patch grid)
function draw() {
  for (let v = 0; v < 4; v++) {
    const g = ctxs[v];
    if (S.imgs[v]) g.drawImage(S.imgs[v].canvas, 0, 0); else { g.fillStyle = '#111'; g.fillRect(0, 0, 512, 288); }
  }
  if (!S.feats || S.mode === 'pixels') return;
  ensureBasis();
  const N = S.gh * S.gw;
  if (S.mode === 'pca') {
    const p0 = S.pc0 || 0;
    if (!S.pcs) { const Y = project(S.basis, S.X, 4 * N, p0 + 3); S.pcs = new Float32Array(4 * N * 3); for (let i = 0; i < 4 * N; i++) for (let c = 0; c < 3; c++) S.pcs[i * 3 + c] = Y[i * (p0 + 3) + p0 + c]; }
    const col = toRGB(S.pcs, 4 * N);
    for (let v = 0; v < 4; v++) {
      const g = ctxs[v]; g.globalAlpha = S.alpha;
      for (let r = 0; r < S.gh; r++) for (let c = 0; c < S.gw; c++) {
        const [R, G, B] = col(v * N + r * S.gw + c); g.fillStyle = `rgb(${R | 0},${G | 0},${B | 0})`; g.fillRect(c * 16, r * 16, 16, 16);
      }
      g.globalAlpha = 1;
    }
  } else if (S.mode === 'sim' && S.query) {
    const q = S.query, qi = q.v * N + q.r * S.gw + q.c;
    const sim = simTo(S.normed, 4 * N, S.D, qi);
    const mxs = viewMax(sim, N, q.v, qi);
    for (let v = 0; v < 4; v++) {
      const g = ctxs[v], mx = mxs[v];
      g.fillStyle = `rgba(0,0,0,${S.alpha * 0.75})`; g.fillRect(0, 0, 512, 288);
      for (let r = 0; r < S.gh; r++) for (let c = 0; c < S.gw; c++) {
        const s = sim[v * N + r * S.gw + c] / mx; if (s < 0.25) continue;
        const [R, G, B] = viridis((s - 0.25) / 0.75);
        g.fillStyle = `rgba(${R | 0},${G | 0},${B | 0},${S.alpha * Math.min(1, (s - 0.2) * 1.6)})`; g.fillRect(c * 16, r * 16, 16, 16);
      }
      if (v === q.v) { g.strokeStyle = '#fff'; g.lineWidth = 2.5; g.strokeRect(q.c * 16 + 1, q.r * 16 + 1, 14, 14); }
    }
  }
}

// ------------------------------------------------------------------ bottleneck demo
const latGrid = $('#latGrid'); const latC = [];
for (let v = 0; v < 4; v++) { const c = document.createElement('canvas'); c.width = 512; c.height = 288; latGrid.appendChild(c); latC.push(c.getContext('2d')); }
$('#chan').oninput = (e) => { S.latChan = +e.target.value; $('#chanV').textContent = S.latChan; drawLatent(); };
$('#pool').onchange = (e) => { S.pool = e.target.checked; S.latBasis = null; drawLatent(); };
$$('#latMode button').forEach((b) => (b.onclick = () => { S.latMode = b.dataset.m; $$('#latMode button').forEach((x) => x.classList.toggle('on', x === b)); if (S.latMode === 'sim' && !S.query) S.query = { v: 0, r: 9, c: 16 }; drawLatent(); }));

function pooled() {
  const D = S.D, gh = S.gh, gw = S.gw;
  if (!S.pool) return { X: stackFeats(S.feats, gh * gw, D), h: gh, w: gw };
  const h = gh / 2, w = gw / 2, N = h * w, X = new Float32Array(4 * N * D);
  S.feats.forEach((f, v) => {
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      const o = (v * N + r * w + c) * D;
      for (const [dr, dc] of [[0, 0], [0, 1], [1, 0], [1, 1]]) { const io = ((2 * r + dr) * gw + 2 * c + dc) * D; for (let d = 0; d < D; d++) X[o + d] += f[io + d] / 4; }
    }
  });
  return { X, h, w };
}

function drawLatent() {
  const cellsEl = $('#cells'), numsEl = $('#nums');
  const h = S.pool ? 9 : 18, w = S.pool ? 16 : 32;
  cellsEl.innerHTML = `${h}<span class="muted" style="font-size:12px">(rows)</span>×${w}<span class="muted" style="font-size:12px">(cols)</span> = ${h * w}`;
  numsEl.textContent = (h * w * S.latChan).toLocaleString();
  $('#ratio').textContent = `${((288 * 512 * 3) / (h * w * S.latChan)).toFixed(0)}×`; $('#ratio2').textContent = `${((2 * 288 * 512 * 3) / (h * w * S.latChan)).toFixed(0)}×`;
  for (const g of latC) { g.fillStyle = '#0d1118'; g.fillRect(0, 0, 512, 288); g.fillStyle = '#66728a'; g.font = '14px sans-serif'; g.fillText(S.feats ? '' : 'encode a frame in Demo 1 first', 150, 148); }
  if (!S.feats) { $('#varKept').textContent = '–'; return; }
  if (!S.latBasis) { const p = pooled(); S.latP = p; S.latBasis = pca(p.X, 4 * p.h * p.w, S.D); }
  const { X, h: H, w: W } = S.latP, N = H * W, k = S.latChan, B = S.latBasis;
  let kept = 0; for (let i = 0; i < k; i++) kept += Math.max(0, B.values[i]);
  $('#varKept').textContent = ((100 * kept) / B.total).toFixed(1) + '%';
  const Z = project(B, X, 4 * N, k); // the latent: 4 views × N cells × k channels
  const cw = 512 / W, chh = 288 / H;
  if (S.latMode === 'pca') {
    // visualise the latent's own top-3 directions (= first 3 channels of a PCA latent)
    const kk = Math.min(3, k), P = new Float32Array(4 * N * 3);
    for (let i = 0; i < 4 * N; i++) for (let c = 0; c < kk; c++) P[i * 3 + c] = Z[i * k + c];
    const col = toRGB(P, 4 * N);
    for (let v = 0; v < 4; v++) for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const [R, G, Bc] = col(v * N + r * W + c); latC[v].fillStyle = `rgb(${R | 0},${G | 0},${Bc | 0})`; latC[v].fillRect(c * cw, r * chh, cw + 0.5, chh + 0.5);
    }
  } else {
    const q = S.query || { v: 0, r: 9, c: 16 };
    const qr = S.pool ? q.r >> 1 : q.r, qc = S.pool ? q.c >> 1 : q.c, qi = q.v * N + qr * W + qc;
    const Zn = normalizeRows(Float32Array.from(Z), 4 * N, k);
    const sim = simTo(Zn, 4 * N, k, qi); const mxs = viewMax(sim, N, q.v, qi);
    for (let v = 0; v < 4; v++) {
      const mx = mxs[v];
      latC[v].drawImage(S.imgs[v].canvas, 0, 0); latC[v].fillStyle = 'rgba(0,0,0,.7)'; latC[v].fillRect(0, 0, 512, 288);
      for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
        const s = sim[v * N + r * W + c] / mx; if (s < 0.25) continue;
        const [R, G, Bc] = viridis((s - 0.25) / 0.75); latC[v].fillStyle = `rgba(${R | 0},${G | 0},${Bc | 0},${Math.min(1, (s - 0.2) * 1.6) * 0.9})`; latC[v].fillRect(c * cw, r * chh, cw, chh);
      }
      if (v === q.v) { latC[v].strokeStyle = '#fff'; latC[v].lineWidth = 2.5; latC[v].strokeRect(qc * cw + 1, qr * chh + 1, cw - 2, chh - 2); }
    }
  }
  // grid lines
  for (const g of latC) { g.strokeStyle = 'rgba(0,0,0,.35)'; g.lineWidth = 1; for (let c = 1; c < W; c++) { g.beginPath(); g.moveTo(c * cw, 0); g.lineTo(c * cw, 288); g.stroke(); } for (let r = 1; r < H; r++) { g.beginPath(); g.moveTo(0, r * chh); g.lineTo(512, r * chh); g.stroke(); } }
}

// ------------------------------------------------------------------ charts + videos
barChart($('#barPSNR'), {
  title: 'Codec reconstruction (PSNR)', better: 'higher', fmt: (v) => v.toFixed(1), max: 34,
  data: [
    { label: 'from scratch', value: 32.2, color: '#fb923c' },
    { label: '+ DINO distill', value: 31.4, color: '#fbbf24' },
    { label: 'frozen DINOv3 (MIRA)', value: 29.7, color: '#4ade80' },
  ],
});
barChart($('#barFID'), {
  title: 'World-model generation (gFID @4s)', better: 'lower', fmt: (v) => v.toFixed(1),
  data: [
    { label: 'pixels (plain)', value: 104.9, color: '#f87171' },
    { label: 'pixels (JiT)', value: 81.0, color: '#f87171' },
    { label: 'from scratch', value: 22.5, color: '#fb923c' },
    { label: '+ DINO distill', value: 15.7, color: '#fbbf24' },
    { label: 'frozen DINOv3 (MIRA)', value: 10.7, color: '#4ade80' },
  ],
});
const vids = $$('.vids video');
$('#vidPlay').onclick = () => {
  const playing = !vids[0].paused;
  if (playing) { vids.forEach((v) => v.pause()); $('#vidPlay').textContent = '▶ Play all in sync'; return; }
  vids.forEach((v) => { v.currentTime = 0; v.play(); }); $('#vidPlay').textContent = '❚❚ Pause';
};
setInterval(() => { if (vids[0].paused) return; const t = vids[0].currentTime; vids.slice(1).forEach((v) => { if (Math.abs(v.currentTime - t) > 0.08) v.currentTime = t; }); }, 500);

buildThumbs(); selectFrame(2);
