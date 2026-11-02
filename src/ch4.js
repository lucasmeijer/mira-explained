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
    space: `<b class="c-wm">Space layer:</b> the white token computes scores against all <b>576</b> tokens of its own frame (bright), in all 4 players' views. It ignores every other frame (dark).`,
    time: `<b class="c-noise">Time layer:</b> the white token (P${S.v + 1}, row ${S.r}, column ${S.c}) only looks at <b>the same cell</b> in its own frame and the ${S.f} frames before it (orange): ${S.f + 1} scores. Later frames are <span style="color:var(--bad)">masked</span>, so it can't see the future. Click a column in the strip to move the query to an older frame.`,
    full: `<b>Naive version:</b> every token looks at every token of every earlier frame, up to <b>11,520</b> scores per query. That works but costs about 20× more than space + time layers together.`,
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
const bp = [0, 1, 2, 3].map(() => Float32Array.from({ length: EP }, () => randn(rng) * 0.6));
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
    parts.push(mv(Wp[p], mh).map((v, i) => silu(v + bp[p][i])));
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
    { name: 'emb(keys)', data: ea, color: '#f472b6' },
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
    sub.split('\n').forEach((t, i) => (svgEl('text', { x: x + w / 2, y: y + 40 + i * 14, fill: '#93a0b4', 'font-size': 11, 'text-anchor': 'middle' }, svg).textContent = t));
  };
  const Y = 130;
  // residual stream line
  svgEl('line', { x1: 20, y1: Y, x2: 1080, y2: Y, stroke: '#a78bfa', 'stroke-width': 4, 'stroke-opacity': 0.6 }, svg);
  svgEl('text', { x: 20, y: Y + 26, fill: '#a78bfa', 'font-size': 11.5 }, svg).textContent = 'residual stream: each token\u2019s 4096 numbers flow straight through…';
  svgEl('text', { x: 1080, y: Y + 26, fill: '#a78bfa', 'font-size': 11.5, 'text-anchor': 'end' }, svg).textContent = '…and each sub-layer adds (+) a correction';
  const subs = [
    [140, '#fbbf24', 'AdaLN → space attention', 'mix tokens within a frame\n(all 4 views)'],
    [470, '#fb923c', 'AdaLN → time attention', 'mix the same cell across\nearlier frames (causal)'],
    [800, '#38bdf8', 'AdaLN → MLP', 'process each token alone\n(like makemore\u2019s hidden layer)'],
  ];
  subs.forEach(([x, c, t, sub]) => {
    box(x, 20, 220, 70, c, t, sub);
    svgEl('path', { d: `M${x - 30},${Y} C ${x - 30},60 ${x - 10},55 ${x},55`, stroke: c, 'stroke-width': 1.6, fill: 'none' }, svg);
    svgEl('path', { d: `M${x + 220},55 C ${x + 250},55 ${x + 260},80 ${x + 260},${Y - 12}`, stroke: c, 'stroke-width': 1.6, fill: 'none' }, svg);
    svgEl('circle', { cx: x + 260, cy: Y, r: 11, fill: '#0b0e14', stroke: c, 'stroke-width': 2 }, svg);
    svgEl('text', { x: x + 260, y: Y + 5, fill: c, 'font-size': 15, 'font-weight': 700, 'text-anchor': 'middle' }, svg).textContent = '+';
  });
  box(380, 185, 340, 52, '#f472b6', 'c = emb(4 players\u2019 keys) + emb(τ)', 'sets γ, β for every AdaLN above (one c per block, same for all tokens)', 0.1);
  [250, 580, 910].forEach((x) => svgEl('path', { d: `M${x},185 L${x},96`, stroke: '#f472b6', 'stroke-width': 1.3, 'stroke-dasharray': '4 3', fill: 'none' }, svg));
}

// ------------------------------------------------------------------ token shape diagram
{
  const svg = $('#tokSvg');
  const T = (x, y, t, o = {}) => (svgEl('text', { x, y, fill: o.fill || '#c8d1de', 'font-size': o.fs || 12, 'text-anchor': o.a || 'start', 'font-weight': o.fw || 400 }, svg).textContent = t);
  const cols = ['#38bdf8', '#38bdf8', '#fb923c', '#fb923c'];
  // four separate grids
  for (let p = 0; p < 4; p++) {
    const x0 = 20 + p * 64, y0 = 40;
    for (let r = 0; r < 9; r++) for (let c = 0; c < 16; c++) svgEl('rect', { x: x0 + c * 3.4, y: y0 + r * 3.4, width: 3, height: 3, fill: cols[p], 'fill-opacity': 0.55 }, svg);
    T(x0 + 27, y0 - 8, `P${p + 1}`, { a: 'middle', fw: 700, fill: cols[p] });
  }
  T(20, 20, 'one moment: 4 (players) × 9 (rows) × 16 (columns), each cell = 32 numbers', { fs: 11.5, fill: '#93a0b4' });
  // arrow
  svgEl('path', { d: 'M150,82 L150,112', stroke: '#5b6b85', 'stroke-width': 2 }, svg); svgEl('path', { d: 'M150,118 l-6,-8 h12 z', fill: '#5b6b85' }, svg);
  T(160, 104, 'stack vertically', { fs: 11.5, fill: '#93a0b4' });
  // tall grid
  const gx = 110, gy = 128, cs = 5;
  for (let p = 0; p < 4; p++) for (let r = 0; r < 9; r++) for (let c = 0; c < 16; c++) svgEl('rect', { x: gx + c * cs, y: gy + (p * 9 + r) * cs, width: cs - 1, height: cs - 1, fill: cols[p], 'fill-opacity': 0.55 }, svg);
  svgEl('rect', { x: gx + 5 * cs - 1, y: gy + 13 * cs - 1, width: cs + 1, height: cs + 1, fill: 'none', stroke: '#fff', 'stroke-width': 1.5 }, svg);
  T(gx + 40, gy + 36 * cs + 18, '36 (rows) × 16 (columns)', { a: 'middle', fs: 11.5 });
  T(gx + 40, gy + 36 * cs + 33, '= 576 tokens per frame', { a: 'middle', fs: 11.5, fill: '#fbbf24', fw: 700 });
  // one token zoom
  const zx = 250, zy = 170;
  svgEl('path', { d: `M${gx + 6 * cs},${gy + 13.5 * cs} L${zx - 6},${zy + 20}`, stroke: '#fff', 'stroke-width': 1, 'stroke-dasharray': '3 3' }, svg);
  for (let i = 0; i < 32; i++) svgEl('rect', { x: zx + i * 3, y: zy + 10 - Math.sin(i * 1.7) * 6, width: 2.2, height: 12 + Math.sin(i * 1.7) * 6, fill: '#a78bfa' }, svg);
  T(zx, zy - 4, 'one token = 32 (channels) numbers', { fs: 11.5, fill: '#a78bfa', fw: 700 });
  T(zx, zy + 44, 'linear layer 32 → 4096 (like makemore\u2019s C)', { fs: 11.5, fill: '#93a0b4' });
  svgEl('path', { d: `M${zx + 48},${zy + 52} L${zx + 48},${zy + 72}`, stroke: '#5b6b85', 'stroke-width': 2 }, svg);
  for (let i = 0; i < 96; i++) svgEl('rect', { x: zx + i * 3, y: zy + 86 - Math.cos(i * 0.9) * 5, width: 2, height: 8 + Math.cos(i * 0.9) * 5, fill: '#fbbf24', 'fill-opacity': 0.8 }, svg);
  T(zx, zy + 118, '4096 (features) numbers: the token\u2019s vector', { fs: 11.5, fill: '#fbbf24', fw: 700 });
  T(zx, zy + 134, 'refined by 16 blocks (drawn: 96 of 4096)', { fs: 11, fill: '#93a0b4' });
  // time
  T(250, 345, '× 20 (frames of context, 2 s) → 11,520 tokens', { fs: 12, fw: 700 });
  for (let f = 0; f < 20; f++) svgEl('rect', { x: 250 + f * 14, y: 356, width: 11, height: 26, rx: 2, fill: '#fbbf24', 'fill-opacity': 0.15 + f * 0.03, stroke: '#fbbf24', 'stroke-opacity': 0.4 }, svg);
  T(250, 398, 'oldest', { fs: 10.5, fill: '#66728a' }); T(527, 398, 'newest (being generated)', { fs: 10.5, fill: '#66728a', a: 'end' });
}

// ------------------------------------------------------------------ attention-by-hand demo
{
  const hero = await loadImage('/frames/hero/f_0100.jpg');
  const DIMS = ['ball', 'car', 'stadium/sky', 'clock'];
  const TOK = [
    { name: 'P1 · ball', at: [235, 175], x: [1.0, 0.45, 0.0, 0.0] },
    { name: 'P2 · ball', at: [750, 92], x: [0.95, 0.0, 0.15, 0.0] },
    { name: 'P3 · ball', at: [272, 383], x: [0.9, 0.0, 0.25, 0.0] },
    { name: 'P3 · car', at: [262, 500], x: [0.1, 1.0, 0.0, 0.0] },
    { name: 'P1 · stadium', at: [110, 50], x: [0.0, 0.0, 1.0, 0.1] },
    { name: 'P2 · stadium', at: [880, 60], x: [0.0, 0.0, 1.0, 0.0] },
    { name: 'P1 · clock', at: [256, 13], x: [0.0, 0.0, 0.2, 1.0] },
    { name: 'P4 · clock', at: [768, 301], x: [0.0, 0.0, 0.15, 1.0] },
  ];
  let qi = 0, sharp = 1.4;
  const cards = $('#cards');
  TOK.forEach((t, i) => {
    const d = document.createElement('div'); d.className = 'card';
    const c = document.createElement('canvas'); c.width = 96; c.height = 64;
    c.getContext('2d').drawImage(hero, t.at[0] - 36, t.at[1] - 24, 72, 48, 0, 0, 96, 64);
    d.append(c); d.insertAdjacentHTML('beforeend', `<div>${t.name}</div><div class="w"><div></div></div>`);
    d.onclick = () => { qi = i; draw(); };
    cards.appendChild(d); t.el = d;
  });
  $('#sharp').oninput = (e) => { sharp = +e.target.value; draw(); };
  const cv = $('#cvAtt'), g = cv.getContext('2d');
  function draw() {
    $('#sharpV').textContent = sharp.toFixed(2);
    const q = TOK[qi].x, d = q.length;
    const scores = TOK.map((t) => (sharp * 4 * t.x.reduce((s, v, i) => s + v * q[i], 0)) / Math.sqrt(d));
    const mx = Math.max(...scores), ex = scores.map((s) => Math.exp(s - mx)), Z = ex.reduce((a, b) => a + b, 0), w = ex.map((e) => e / Z);
    const out = q.map((_, k) => TOK.reduce((s, t, j) => s + w[j] * t.x[k], 0));
    TOK.forEach((t, i) => { t.el.classList.toggle('q', i === qi); t.el.querySelector('.w > div').style.width = `${w[i] * 100}%`; });
    g.fillStyle = '#0a0d13'; g.fillRect(0, 0, cv.width, cv.height);
    g.font = '600 11.5px ui-monospace,monospace';
    g.fillStyle = '#93a0b4'; g.fillText('token', 10, 18); g.fillText('q·k/√d', 150, 18); g.fillText('softmax weight', 240, 18);
    TOK.forEach((t, i) => {
      const y = 36 + i * 25;
      g.fillStyle = i === qi ? '#fff' : '#c8d1de'; g.fillText(t.name, 10, y);
      g.fillStyle = '#fb923c'; g.fillText(scores[i].toFixed(2), 150, y);
      g.fillStyle = '#fbbf24'; g.fillRect(240, y - 10, w[i] * 150, 12);
      g.fillStyle = '#e6ebf2'; g.fillText(`${(w[i] * 100).toFixed(0)}%`, 245 + w[i] * 150, y);
    });
    // vectors
    const bx = 440;
    g.fillStyle = '#93a0b4'; g.fillText('query token x', bx, 18); g.fillText('out = Σ w·v', bx + 95, 18);
    DIMS.forEach((dn, k) => {
      const y = 44 + k * 48;
      g.fillStyle = '#66728a'; g.font = '11px Inter, sans-serif'; g.fillText(dn, bx, y - 8); g.font = '600 11.5px ui-monospace,monospace';
      g.fillStyle = '#5b6b85'; g.fillRect(bx, y, q[k] * 70, 12);
      g.fillStyle = '#38bdf8'; g.fillRect(bx + 95, y, out[k] * 60, 12);
      g.fillStyle = '#e6ebf2'; g.fillText(out[k].toFixed(2), bx + 99 + out[k] * 60, y + 10);
    });
    g.fillStyle = '#93a0b4'; g.font = '11px Inter, sans-serif';
    g.fillText(`→ ${TOK[qi].name} gathers mostly from tokens with similar content, whichever player's view they're in.`, 10, cv.height - 8);
  }
  draw();
}
