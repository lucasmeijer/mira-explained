import { mountNav, $, $$ } from './nav.js';
import { loadImage } from './lib/dino.js';
import { mulberry32, randn } from './lib/nn.js';
import { svgEl } from './lib/charts.js';
mountNav('ch4-transformer');

// ------------------------------------------------------------------ token layout explorer
const NF = 20, GH = 9, GW = 16;
const imgs = await Promise.all(Array.from({ length: NF }, (_, k) => loadImage(`/frames/seq/s_${String(30 + k * 2).padStart(3, '0')}.jpg`)));
const Q = [[0, 0], [256, 0], [0, 144], [256, 144]]; // quadrant of each player in the 2x2 source frames
const S = { mode: 'space', v: 0, r: 4, c: 8, f: NF - 1 };
const cvF = $('#cvFrame'), gF = cvF.getContext('2d');
const cvS = $('#cvStrip'), gS = cvS.getContext('2d');
cvS.height = 470;
const BIG = { w: 300, vh: 300 * 9 / 16, gap: 4 };
const ST = { x0: 4, y0: 22, colW: 85.5, vw: 81, vh: 45.5, gap: 3, rowH: 222, perRow: 10 };
const colX = (f) => ST.x0 + (f % ST.perRow) * ST.colW, rowY = (f) => Math.floor(f / ST.perRow) * ST.rowH;
const AMB = 'rgba(251,191,36,', ORA = 'rgba(251,146,60,';

function drawBig() {
  gF.fillStyle = '#0a0d13'; gF.fillRect(0, 0, cvF.width, cvF.height);
  const im = imgs[S.f];
  for (let v = 0; v < 4; v++) {
    const y0 = v * (BIG.vh + BIG.gap);
    gF.drawImage(im, Q[v][0], Q[v][1], 256, 144, 0, y0, BIG.w, BIG.vh);
    const cw = BIG.w / GW, ch = BIG.vh / GH;
    for (let r = 0; r < GH; r++) for (let c = 0; c < GW; c++) {
      const isQ = v === S.v && r === S.r && c === S.c;
      let a = 0.55; // dim
      if (S.mode === 'space' || S.mode === 'full') a = 0.12;
      if (isQ) a = 0;
      gF.fillStyle = `rgba(10,13,19,${a})`; gF.fillRect(c * cw, y0 + r * ch, cw, ch);
      if ((S.mode === 'space' || S.mode === 'full') && !isQ) { gF.fillStyle = AMB + '0.16)'; gF.fillRect(c * cw, y0 + r * ch, cw, ch); }
    }
    gF.strokeStyle = 'rgba(0,0,0,0.35)'; gF.lineWidth = 1;
    for (let c = 1; c < GW; c++) { gF.beginPath(); gF.moveTo(c * cw, y0); gF.lineTo(c * cw, y0 + BIG.vh); gF.stroke(); }
    for (let r = 1; r < GH; r++) { gF.beginPath(); gF.moveTo(0, y0 + r * ch); gF.lineTo(BIG.w, y0 + r * ch); gF.stroke(); }
    gF.fillStyle = 'rgba(0,0,0,.65)'; gF.fillRect(4, y0 + 4, 26, 15); gF.fillStyle = '#e6ebf2'; gF.font = '700 10px ui-monospace,monospace'; gF.fillText(`P${v + 1}`, 9, y0 + 15);
    if (v === S.v) { gF.strokeStyle = '#fff'; gF.lineWidth = 2.5; gF.strokeRect(S.c * cw + 1, y0 + S.r * ch + 1, cw - 2, ch - 2); }
  }
}
function drawStrip() {
  gS.fillStyle = '#0a0d13'; gS.fillRect(0, 0, cvS.width, cvS.height);
  gS.font = '600 10px ui-monospace,monospace';
  for (let f = 0; f < NF; f++) {
    const x = colX(f), oy = rowY(f), future = f > S.f;
    gS.fillStyle = f === S.f ? '#fbbf24' : '#66728a';
    gS.fillText(f === S.f ? 't (query)' : f < S.f ? `t${f - S.f}` : `t+${f - S.f}`, x + 2, oy + 15);
    for (let v = 0; v < 4; v++) {
      const y = oy + ST.y0 + v * (ST.vh + ST.gap);
      gS.drawImage(imgs[f], Q[v][0], Q[v][1], 256, 144, x, y, ST.vw, ST.vh);
      let shade = 0.65;
      if (S.mode === 'space' && f === S.f) shade = 0.05;
      if (S.mode === 'full' && !future) shade = 0.05;
      if (S.mode === 'time') shade = 0.6;
      if (future) shade = 0.88;
      gS.fillStyle = `rgba(10,13,19,${shade})`; gS.fillRect(x, y, ST.vw, ST.vh);
      if (S.mode === 'space' && f === S.f) { gS.fillStyle = AMB + '0.22)'; gS.fillRect(x, y, ST.vw, ST.vh); }
      if (S.mode === 'full' && !future) { gS.fillStyle = AMB + '0.14)'; gS.fillRect(x, y, ST.vw, ST.vh); }
      const cw = ST.vw / GW, ch = ST.vh / GH;
      if (S.mode === 'time' && v === S.v && !future) { gS.fillStyle = ORA + '1)'; gS.fillRect(x + S.c * cw - 1, y + S.r * ch - 1, cw + 2, ch + 2); }
      if (f === S.f && v === S.v) { gS.strokeStyle = '#fff'; gS.lineWidth = 1.5; gS.strokeRect(x + S.c * cw - 1.5, y + S.r * ch - 1.5, cw + 3, ch + 3); }
    }
    if (future) { gS.fillStyle = '#f87171'; gS.font = '700 11px ui-monospace,monospace'; gS.fillText('MASKED', x + 18, oy + ST.y0 + 2 * (ST.vh + ST.gap)); gS.font = '600 10px ui-monospace,monospace'; }
    if (f === S.f) { gS.strokeStyle = '#fbbf24'; gS.lineWidth = 2; gS.strokeRect(x - 2, oy + ST.y0 - 2, ST.vw + 4, 4 * (ST.vh + ST.gap) + 1); }
  }
  gS.fillStyle = '#93a0b4'; gS.font = '11px Inter, sans-serif';
  gS.fillText('context window: 20 latent frames = 2 s at 10 Hz, each = 4 stacked player views · click a column to move the query frame', ST.x0, cvS.height - 8);
  const notes = {
    space: `<b class="c-wm">Space layer:</b> the query token (white) attends to all <b>576</b> tokens of its own frame, in all 4 views at once. Bidirectional inside the frame, nothing from other frames.`,
    time: `<b class="c-noise">Time layer:</b> the query attends only to the <b>same cell</b> (P${S.v + 1}, row ${S.r}, col ${S.c}) in frame t and the ${S.f} frames before it, ${S.f + 1} keys in total. Future frames are masked (causal).`,
    full: `<b>Naive full attention</b> would let every token see every token of every past frame: up to <b>11,520</b> keys per query. Splitting it into space + time layers cuts the pair count by about 20×.`,
  };
  $('#attnNote').innerHTML = notes[S.mode];
}
function redraw() { drawBig(); drawStrip(); }
cvF.addEventListener('mousemove', pick); cvF.addEventListener('click', pick);
function pick(e) {
  const r = cvF.getBoundingClientRect(), x = ((e.clientX - r.left) / r.width) * cvF.width, y = ((e.clientY - r.top) / r.height) * cvF.height;
  const v = Math.min(3, Math.floor(y / (BIG.vh + BIG.gap))), yy = y - v * (BIG.vh + BIG.gap);
  if (yy > BIG.vh) return;
  S.v = v; S.r = Math.min(GH - 1, Math.floor(yy / (BIG.vh / GH))); S.c = Math.min(GW - 1, Math.floor(x / (BIG.w / GW)));
  redraw();
}
cvS.addEventListener('click', (e) => {
  const r = cvS.getBoundingClientRect(), x = ((e.clientX - r.left) / r.width) * cvS.width;
  const y = ((e.clientY - r.top) / r.height) * cvS.height; const row = Math.min(1, Math.floor(y / ST.rowH)); S.f = Math.max(0, Math.min(NF - 1, row * ST.perRow + Math.floor((x - ST.x0) / ST.colW))); redraw();
});
$$('#attnMode button').forEach((b) => (b.onclick = () => { S.mode = b.dataset.m; $$('#attnMode button').forEach((x) => x.classList.toggle('on', x === b)); redraw(); }));
redraw();

// ------------------------------------------------------------------ AdaLN demo (random weights, real wiring)
const KEYS = ['Q', 'W', 'E', 'A', 'S', 'D', 'Shift', 'Space', 'Ctrl'];
const KEY_IDX = { Q: 4, W: 0, E: 5, A: 2, S: 1, D: 3, Shift: 7, Space: 6, Ctrl: 8 }; // paper order: W A S D Q E SPACE LSHIFT LCTRL
const rng = mulberry32(42);
const mat = (r, c, s = 1) => Array.from({ length: r }, () => Float32Array.from({ length: c }, () => randn(rng) * s / Math.sqrt(c)));
const EP = 12, DC = 48, DX = 32;
const Wp = [0, 1, 2, 3].map(() => mat(EP, 9, 1.6));          // per-player key embedding
const absentTok = [0, 1, 2, 3].map(() => Float32Array.from({ length: EP }, () => randn(rng) * 0.8));
const Wa = mat(DC, 4 * EP, 1.4);                                // mixes the concatenated players into c
const Wt1 = mat(DC, 16, 1.5), Wt2 = mat(DC, DC, 1.2);             // tau MLP
const Wg = mat(DX, DC, 0.5), Wb = mat(DX, DC, 0.7);               // c -> gamma, beta
const tokens = [0, 1, 2, 3].map(() => Float32Array.from({ length: DX }, () => randn(rng) * 1.3 + 0.4));
const st = { keys: [0, 1, 2, 3].map(() => new Set()), absent: [false, false, false, false], tau: 0.5, tok: 0 };
const mv = (M, x) => Float32Array.from(M, (row) => row.reduce((s, w, i) => s + w * x[i], 0));
const silu = (x) => x / (1 + Math.exp(-x));

const kb = $('#kb');
for (let p = 0; p < 4; p++) {
  const d = document.createElement('div'); d.className = 'pl';
  d.innerHTML = `<h4 style="color:${['#38bdf8', '#38bdf8', '#fb923c', '#fb923c'][p]}">Player ${p + 1} <span class="tag">(${p < 2 ? 'blue' : 'orange'})</span></h4><div class="keys"></div><label class="absent"><input type="checkbox"/> absent (action dropout)</label>`;
  const kk = d.querySelector('.keys');
  KEYS.forEach((k) => { const b = document.createElement('button'); b.textContent = k; b.onclick = () => { st.keys[p].has(k) ? st.keys[p].delete(k) : st.keys[p].add(k); b.classList.toggle('down'); compute(); }; kk.appendChild(b); });
  d.querySelector('input').onchange = (e) => { st.absent[p] = e.target.checked; d.querySelector('.keys').style.opacity = e.target.checked ? 0.3 : 1; compute(); };
  kb.appendChild(d);
}
$('#adaTau').oninput = (e) => { st.tau = +e.target.value; $('#adaTauV').textContent = st.tau.toFixed(2); compute(); };
$('#tokSel').onchange = (e) => { st.tok = +e.target.value; compute(); };
$('#randKeys').onclick = () => {
  $$('#kb .keys').forEach((kk, p) => { st.keys[p].clear(); [...kk.children].forEach((b) => { const on = Math.random() < 0.25; b.classList.toggle('down', on); if (on) st.keys[p].add(b.textContent); }); });
  compute();
};

function bars(cv, series, { min = -2.5, max = 2.5, labels = [] } = {}) {
  const g = cv.getContext('2d'), W = cv.width, H = cv.height;
  g.fillStyle = '#0a0d13'; g.fillRect(0, 0, W, H);
  const rowH = (H - 6) / series.length;
  series.forEach((s, k) => {
    const y0 = 3 + k * rowH, mid = y0 + rowH / 2, n = s.data.length, bw = (W - 70) / n;
    g.strokeStyle = '#263041'; g.beginPath(); g.moveTo(66, mid); g.lineTo(W, mid); g.stroke();
    g.fillStyle = '#93a0b4'; g.font = '600 10.5px ui-monospace,monospace'; g.fillText(s.name, 4, mid + 4);
    for (let i = 0; i < n; i++) {
      const v = Math.max(min, Math.min(max, s.data[i])), h = (v / (max - min)) * (rowH - 6);
      g.fillStyle = s.color; g.globalAlpha = s.alpha ?? 0.9; g.fillRect(68 + i * bw, mid - Math.max(0, h), Math.max(1, bw - 1), Math.abs(h)); g.globalAlpha = 1;
    }
  });
}
function compute() {
  // per-player embeddings, concatenated in fixed player order
  const parts = [];
  for (let p = 0; p < 4; p++) {
    if (st.absent[p]) { parts.push(absentTok[p]); continue; }
    const mh = new Float32Array(9); st.keys[p].forEach((k) => (mh[KEY_IDX[k]] = 1));
    parts.push(mv(Wp[p], mh).map(silu));
  }
  const cat = new Float32Array(4 * EP); parts.forEach((v, p) => cat.set(v, p * EP));
  const ea = mv(Wa, cat);
  const sin = Float32Array.from({ length: 16 }, (_, i) => (i < 8 ? Math.sin(st.tau * Math.pow(2, i) * 3) : Math.cos(st.tau * Math.pow(2, i - 8) * 3)));
  const et = mv(Wt2, mv(Wt1, sin).map(silu));
  const c = ea.map((v, i) => v + et[i]);
  const gam = mv(Wg, c.map(silu)), bet = mv(Wb, c.map(silu));
  const x = tokens[st.tok], mu = x.reduce((a, b) => a + b, 0) / DX, sd = Math.sqrt(x.reduce((a, b) => a + (b - mu) ** 2, 0) / DX);
  const ln = x.map((v) => (v - mu) / sd), out = ln.map((v, i) => (1 + gam[i]) * v + bet[i]);
  bars($('#cvC'), [
    { name: 'emb(a)', data: ea, color: '#f472b6' },
    { name: 'emb(τ)', data: et, color: '#fb923c' },
    { name: 'c', data: c, color: '#fbbf24' },
  ], { min: -4, max: 4 });
  bars($('#cvGB'), [{ name: 'γ', data: gam, color: '#fbbf24' }, { name: 'β', data: bet, color: '#fbbf24', alpha: 0.6 }], { min: -3, max: 3 });
  bars($('#cvX'), [{ name: 'LN(x)', data: ln, color: '#93a0b4' }, { name: 'out', data: out, color: '#38bdf8' }], { min: -4, max: 4 });
}
compute();

// ------------------------------------------------------------------ block diagram
{
  const svg = $('#blockSvg');
  const box = (x, y, w, h, color, title, sub, fill = 0.12) => {
    svgEl('rect', { x, y, width: w, height: h, rx: 10, fill: color, 'fill-opacity': fill, stroke: color, 'stroke-width': 1.5 }, svg);
    svgEl('text', { x: x + w / 2, y: y + 22, fill: '#e6ebf2', 'font-size': 13, 'font-weight': 700, 'text-anchor': 'middle' }, svg).textContent = title;
    sub.split('\n').forEach((s, i) => (svgEl('text', { x: x + w / 2, y: y + 40 + i * 14, fill: '#93a0b4', 'font-size': 11, 'text-anchor': 'middle' }, svg).textContent = s));
  };
  const arr = (x1, x2, y) => { svgEl('line', { x1, y1: y, x2: x2 - 6, y2: y, stroke: '#5b6b85', 'stroke-width': 1.8 }, svg); svgEl('path', { d: `M${x2},${y} l-8,-5 v10 z`, fill: '#5b6b85' }, svg); };
  box(10, 50, 120, 80, '#a78bfa', 'z_τ tokens', '20 frames × 576\n(+ registers)', 0.2);
  arr(130, 165, 90);
  box(165, 40, 220, 100, '#fbbf24', 'Space attention', 'within a frame, all 4 views\nbidirectional · GQA · gated\nQK-norm · spatial RoPE');
  arr(385, 420, 90);
  box(420, 40, 220, 100, '#fb923c', 'Time attention', 'same cell across frames\ncausal · GQA · gated\nQK-norm · temporal RoPE (s)');
  arr(640, 675, 90);
  box(675, 40, 220, 100, '#38bdf8', 'AdaLN · SwiGLU MLP', '(1+γ)·LN(x)+β → SwiGLU\nper-token, no mixing');
  arr(895, 930, 90);
  box(930, 50, 160, 80, '#a78bfa', 'v_θ', 'linear head → velocity\nsplit back into 4 views', 0.2);
  // conditioning
  box(560, 170, 230, 52, '#f472b6', 'c = emb(actions) + emb(τ)', '', 0.12);
  svgEl('path', { d: 'M785,170 C 785,150 785,150 785,140', stroke: '#f472b6', 'stroke-width': 1.6, 'stroke-dasharray': '4 3', fill: 'none' }, svg);
  svgEl('text', { x: 795, y: 160, fill: '#f472b6', 'font-size': 11 }, svg).textContent = 'γ, β (once per block, broadcast to all tokens)';
  svgEl('text', { x: 165, y: 175, fill: '#66728a', 'font-size': 11 }, svg).textContent = 'Past latents enter only through attention (KV cache at inference).';
  svgEl('text', { x: 165, y: 192, fill: '#66728a', 'font-size': 11 }, svg).textContent = 'Actions & τ enter only through AdaLN.';
  svgEl('text', { x: 10, y: 22, fill: '#93a0b4', 'font-size': 12, 'font-weight': 700, 'letter-spacing': '0.08em' }, svg).textContent = '× 16 BLOCKS';
}
