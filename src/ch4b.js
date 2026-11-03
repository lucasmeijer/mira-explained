import { mountNav, $, $$ } from './nav.js';
import { loadImage } from './lib/dino.js';
import { svgEl } from './lib/charts.js';
mountNav('ch4b-attention');

// ------------------------------------------------------------------ data: real MIRA latent cells
const LAT = await (await fetch('/models/toy/latents.json')).json();
const b64 = (s) => { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Float32Array(u.buffer); };
const HERO = ['f_0020', 'f_0060', 'f_0100', 'f_0140', 'f_0180', 'f_0220', 'f_0270', 'f_0320'];
const imgs = await Promise.all(HERO.map((f) => loadImage(`/frames/hero/${f}.jpg`)));
const Z = LAT.views.slice(0, 32).map((v) => b64(v.z)); // 8 frames × 4 views, each 144 cells × 32 channels
// per-channel standardisation (so every channel contributes comparably)
const mu = new Float32Array(32), sd = new Float32Array(32);
Z.forEach((z) => { for (let i = 0; i < 144; i++) for (let c = 0; c < 32; c++) mu[c] += z[i * 32 + c] / (32 * 144); });
Z.forEach((z) => { for (let i = 0; i < 144; i++) for (let c = 0; c < 32; c++) sd[c] += (z[i * 32 + c] - mu[c]) ** 2 / (32 * 144); });
for (let c = 0; c < 32; c++) sd[c] = Math.sqrt(sd[c]) || 1;
const tok = (f, v, r, c) => { const z = Z[f * 4 + v], o = (r * 16 + c) * 32; return Float32Array.from({ length: 32 }, (_, k) => (z[o + k] - mu[k]) / sd[k]); };
// colour for value vectors (channels 1-3), fixed scale
const toRGB = (v) => [0, 1, 2].map((k) => Math.max(0, Math.min(255, 128 + v[k] * 55)));

const VIEW_OFF = [[0, 0], [512, 0], [0, 288], [512, 288]];
const FRAME = 2; // f_0100 for the space layer
const S = {
  mode: 'space', head: 0, scale: true, step: 0, q: 0,
  space: [[0, 5, 7, 'P1 car+ball'], [1, 2, 7, 'P2 far ball'], [2, 2, 8, 'P3 ball'], [2, 6, 8, 'P3 car'], [1, 6, 7, 'P2 car'], [0, 1, 3, 'P1 stadium'], [1, 1, 11, 'P2 stadium'], [0, 0, 8, 'P1 clock'], [3, 0, 8, 'P4 clock'], [3, 7, 8, 'P4 field']],
  timeCell: [0, 5, 7],
};
const STEPS = [
  { k: 'X', t: '① tokens X' }, { k: 'QKV', t: '② project Q, K, V' }, { k: 'S', t: '③ scores Q·Kᵀ' }, { k: 'M', t: '④ mask' },
  { k: 'A', t: '⑤ softmax' }, { k: 'O', t: '⑥ weighted sum A·V' }, { k: 'R', t: '⑦ add back (residual)' },
];
const stepper = $('#stepper');
STEPS.forEach((s, i) => { const b = document.createElement('button'); b.textContent = s.t; b.onclick = () => { S.step = i; render(); }; stepper.appendChild(b); });
const nextBtn = document.createElement('button'); nextBtn.textContent = 'next ▶'; nextBtn.className = 'primary'; nextBtn.onclick = () => { S.step = Math.min(STEPS.length - 1, S.step + 1); render(); }; stepper.appendChild(nextBtn);

function tokens() {
  if (S.mode === 'space') return S.space.map(([v, r, c, name]) => ({ x: tok(FRAME, v, r, c), name, f: FRAME, v, r, c }));
  const [v, r, c] = S.timeCell;
  return HERO.map((_, f) => ({ x: tok(f, v, r, c), name: `t${f + 1}${f === 7 ? ' (now)' : ''}`, f, v, r, c }));
}
function compute() {
  const T = tokens(), n = T.length, d = 8, lo = S.head === 0 ? 8 : 0;
  const Q = T.map((t) => t.x.slice(lo, lo + d)), K = Q.map((q) => q.slice()); // W_Q = W_K = "pick channels" (readable)
  const V = T.map((t) => t.x.slice(0, 3));
  const Sc = Q.map((q) => K.map((k) => q.reduce((s, qi, i) => s + qi * k[i], 0) / (S.scale ? Math.sqrt(d) : 1)));
  const masked = Sc.map((row, i) => row.map((s, j) => (S.mode === 'time' && j > i ? -Infinity : s)));
  const A = masked.map((row) => { const m = Math.max(...row), e = row.map((s) => (s === -Infinity ? 0 : Math.exp(s - m))), z = e.reduce((a, b) => a + b, 0); return e.map((x) => x / z); });
  const O = A.map((row) => [0, 1, 2].map((k) => row.reduce((s, a, j) => s + a * V[j][k], 0)));
  const R = O.map((o, i) => o.map((x, k) => 0.5 * V[i][k] + 0.5 * x)); // illustrative W_O·gate = 0.5
  return { T, n, d, Q, K, V, Sc, masked, A, O, R };
}

// ------------------------------------------------------------------ picker canvas
const pick = $('#cvPick'), gp = pick.getContext('2d');
const PW = 170, PH = 95.6;
function drawPick() {
  gp.fillStyle = '#0a0d13'; gp.fillRect(0, 0, 340, 200);
  const f = S.mode === 'space' ? FRAME : 7;
  for (let v = 0; v < 4; v++) {
    const ox = (v % 2) * PW, oy = Math.floor(v / 2) * (PH + 4);
    gp.drawImage(imgs[f], VIEW_OFF[v][0], VIEW_OFF[v][1], 512, 288, ox, oy, PW, PH);
    gp.fillStyle = 'rgba(0,0,0,.35)'; gp.fillRect(ox, oy, PW, PH);
    gp.fillStyle = '#fff'; gp.font = '700 10px ui-monospace,monospace'; gp.fillText(`P${v + 1}`, ox + 4, oy + 11);
  }
  const cw = PW / 16, ch = PH / 9;
  const mark = (v, r, c, label, hot) => {
    const ox = (v % 2) * PW, oy = Math.floor(v / 2) * (PH + 4);
    gp.strokeStyle = hot ? '#fff' : '#fbbf24'; gp.lineWidth = hot ? 2.5 : 1.5; gp.strokeRect(ox + c * cw, oy + r * ch, cw, ch);
    if (label) { gp.fillStyle = hot ? '#fff' : '#fbbf24'; gp.font = '700 9px ui-monospace,monospace'; gp.fillText(label, ox + c * cw + cw + 1, oy + r * ch + 8); }
  };
  if (S.mode === 'space') S.space.forEach(([v, r, c], i) => mark(v, r, c, String(i + 1), i === S.q));
  else mark(...S.timeCell, 'cell', true);
  $('#pickNote').innerHTML = S.mode === 'space'
    ? `Frame t of the match, 4 players' views. <b>${S.space.length}</b> selected tokens (MIRA would use all 576). Click a cell to add or remove it; click a selected cell to make it the query.`
    : `Time layer: one grid cell (P${S.timeCell[0] + 1}, row ${S.timeCell[1]}, col ${S.timeCell[2]}) followed through 8 frames. Click any cell to follow a different one.`;
}
pick.addEventListener('click', (e) => {
  const r = pick.getBoundingClientRect(), x = ((e.clientX - r.left) / r.width) * 340, y = ((e.clientY - r.top) / r.height) * 200;
  const v = (x >= PW ? 1 : 0) + (y >= PH + 4 ? 2 : 0), ox = (v % 2) * PW, oy = Math.floor(v / 2) * (PH + 4);
  const cc = Math.floor((x - ox) / (PW / 16)), rr = Math.floor((y - oy) / (PH / 9));
  if (cc < 0 || cc > 15 || rr < 0 || rr > 8) return;
  if (S.mode === 'time') { S.timeCell = [v, rr, cc]; S.q = 7; render(); return; }
  const i = S.space.findIndex(([a, b, c]) => a === v && b === rr && c === cc);
  if (i >= 0) { if (S.q === i) { if (S.space.length > 2) { S.space.splice(i, 1); S.q = 0; } } else S.q = i; }
  else if (S.space.length < 12) { S.space.push([v, rr, cc, `P${v + 1} r${rr}c${cc}`]); S.q = S.space.length - 1; }
  render();
});

// ------------------------------------------------------------------ matrix canvas
const cv = $('#cvMat'), g = cv.getContext('2d');
const div = (x) => { const t = Math.max(-1, Math.min(1, x)), a = Math.abs(t), col = t >= 0 ? [251, 191, 36] : [56, 189, 248]; return [18, 22, 30].map((b, i) => b * (1 - a) + col[i] * a); };
function heat(x0, y0, rows, cols, cw, ch, val, title, on, sub) {
  g.globalAlpha = on ? 1 : 0.18;
  g.fillStyle = '#e6ebf2'; g.font = '700 11.5px Inter, sans-serif'; g.fillText(title, x0, y0 - 18);
  if (sub) { g.fillStyle = '#93a0b4'; g.font = '10.5px Inter, sans-serif'; g.fillText(sub, x0, y0 - 5); }
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const c = val(i, j); g.fillStyle = typeof c === 'string' ? c : `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
    g.fillRect(x0 + j * cw, y0 + i * ch, cw - 1, ch - 1);
  }
  g.globalAlpha = 1;
}
let layout = {};
function drawMat(R) {
  const { T, n, Q, K, V, Sc, masked, A, O, R: Res } = R;
  g.fillStyle = '#0a0d13'; g.fillRect(0, 0, cv.width, cv.height);
  const ch = Math.min(20, Math.floor(250 / n)), y0 = 52, st = S.step;
  // row labels with crops
  T.forEach((t, i) => {
    const y = y0 + i * ch;
    if (i === S.q) { g.fillStyle = 'rgba(255,255,255,.08)'; g.fillRect(0, y - 1, cv.width, ch); }
    g.drawImage(imgs[t.f], VIEW_OFF[t.v][0] + t.c * 32, VIEW_OFF[t.v][1] + t.r * 32, 32, 32, 2, y, ch - 1, ch - 1);
    g.fillStyle = i === S.q ? '#fff' : '#c8d1de'; g.font = `${i === S.q ? 700 : 500} 10.5px Inter, sans-serif`;
    g.fillText(t.name, ch + 6, y + ch * 0.7);
  });
  let x = 92;
  const cwX = 3.6; heat(x, y0, n, 32, cwX, ch, (i, j) => div(T[i].x[j] / 2.2), 'X  tokens', st >= 0, `n × 32 ch`); layout.X = x; x += 32 * cwX + 16;
  const cwQ = 9; heat(x, y0, n, 8, cwQ, ch, (i, j) => div(Q[i][j] / 2.2), 'Q', st >= 1, 'n × 8'); x += 8 * cwQ + 8;
  heat(x, y0, n, 8, cwQ, ch, (i, j) => div(K[i][j] / 2.2), 'K', st >= 1, 'n × 8'); x += 8 * cwQ + 16;
  const cwS = Math.min(16, Math.floor(180 / n));
  const smax = Math.max(...Sc.flat().map(Math.abs)) || 1;
  const showMask = st >= 3;
  heat(x, y0, n, n, cwS, ch, (i, j) => (showMask && masked[i][j] === -Infinity ? '#2a0f14' : div((st >= 3 ? masked : Sc)[i][j] / smax)), st >= 3 ? 'S + mask' : 'S = Q·Kᵀ/√d', st >= 2, 'n × n');
  layout.S = x; x += n * cwS + 16;
  heat(x, y0, n, n, cwS, ch, (i, j) => { const a = A[i][j]; return [251 * a + 18 * (1 - a), 191 * a + 22 * (1 - a), 36 * a + 30 * (1 - a)]; }, 'A = softmax(S)', st >= 4, 'rows sum to 1');
  layout.A = x; x += n * cwS + 16;
  heat(x, y0, n, 1, 22, ch, (i) => toRGB(V[i]), 'V', st >= 1, 'colour'); x += 22 + 18;
  heat(x, y0, n, 1, 22, ch, (i) => toRGB(O[i]), 'O', st >= 5, 'A·V'); x += 22 + 18;
  heat(x, y0, n, 1, 22, ch, (i) => toRGB(Res[i]), 'x+O', st >= 6, 'resid.');
  // highlight query row in S and A
  g.strokeStyle = '#fff'; g.lineWidth = 1.5;
  if (st >= 2) g.strokeRect(layout.S - 1, y0 + S.q * ch - 1, n * cwS + 1, ch + 1);
  if (st >= 4) g.strokeRect(layout.A - 1, y0 + S.q * ch - 1, n * cwS + 1, ch + 1);
  // weights of the query row as bars at bottom
  if (st >= 4) {
    const by = y0 + n * ch + 22; g.fillStyle = '#93a0b4'; g.font = '11px Inter, sans-serif';
    g.fillText(`attention weights of "${T[S.q].name}":`, 92, by);
    const order = A[S.q].map((a, j) => [a, j]).sort((a, b) => b[0] - a[0]).slice(0, 4);
    order.forEach(([a, j], k) => { g.fillStyle = '#fbbf24'; g.fillRect(320 + k * 140, by - 10, a * 70, 10); g.fillStyle = '#e6ebf2'; g.fillText(`${T[j].name} ${(a * 100).toFixed(0)}%`, 320 + k * 140 + a * 70 + 4, by); });
  }
  layout.ch = ch; layout.y0 = y0;
}
cv.addEventListener('click', (e) => {
  const r = cv.getBoundingClientRect(), y = ((e.clientY - r.top) / r.height) * cv.height;
  const i = Math.floor((y - layout.y0) / layout.ch), n = tokens().length;
  if (i >= 0 && i < n) { S.q = i; render(); }
});

function explain(R) {
  const { T, n, Sc, A, masked } = R, q = S.q, name = T[q].name;
  const best = A[q].map((a, j) => [a, j]).sort((a, b) => b[0] - a[0]);
  const top = best.filter(([, j]) => j !== q)[0];
  const space = S.mode === 'space';
  const txt = [
    [`Tokens: one row per token`, `Each row is one latent cell: <b>32 numbers</b> describing a 32×32-pixel patch of one player's screen, coloured blue (negative) to amber (positive). This demo shows ${n} tokens.`,
      `In MIRA each token is first embedded to <b>4096</b> numbers, and a ${space ? 'space layer sees all <b>576</b> tokens of a frame (4 views × 9 × 16)' : 'time layer sees the same cell in up to <b>20</b> frames'}. Before attention, AdaLN normalises each token and applies the keys-dependent γ, β.`],
    [`Project: queries, keys, values`, `Three matrix multiplies give every token a <b class="c-wm">query</b>, a <b class="c-noise">key</b> and a <b class="c-dec">value</b>. Here W<sub>Q</sub> = W<sub>K</sub> just picks latent channels ${S.head === 0 ? '9–16' : '1–8'} (head ${S.head + 1}), so "looking for" and "offering" use the same features. The values are channels 1–3, drawn as one colour per token.`,
      `In MIRA: X is 576 × 4096; W<sub>Q</sub> is 4096 × 4096 (32 heads × 128), W<sub>K</sub>, W<sub>V</sub> are 4096 × 1024 (8 shared heads × 128). All learned. QK-norm then normalises q and k, and RoPE rotates them by position (§4).`],
    [`Scores: every query against every key`, `S[i][j] = q<sub>i</sub>·k<sub>j</sub> ${S.scale ? '/ √8' : '(no scaling)'}: one number per pair. Bright amber means similar. The white row is <b>${name}</b>: its highest score (besides itself) is with <b>${T[top[1]].name}</b> (${Sc[q][top[1]].toFixed(2)}).`,
      `Dividing by √d keeps scores from growing with the head size. Untick "divide by √d" and watch the softmax in step ⑤ become much spikier. In MIRA one head's score matrix in a space layer is 576 × 576 = 331,776 numbers, ×32 heads, ×20 frames.`],
    [`Mask: what a token may <i>not</i> see`, space ? `<b>Space layers use no mask</b>. Every token in the frame may look at every other, in all four views, past and future positions alike, because they are all the same moment.` : `<b>Time layers are causal</b>: frame t may only look at frames ≤ t. Scores above the diagonal are set to −∞ (dark red), which becomes exactly 0 weight after softmax. This is what stops the model from peeking at the future it has to predict.`,
      `In MIRA: space = full attention within a frame; time = causal across frames (the same rule lets the KV cache work while streaming).`],
    [`Softmax: scores → weights`, `Each row is exponentiated and normalised to sum to 1. <b>${name}</b> gives ${(best[0][0] * 100).toFixed(0)}% to <b>${T[best[0][1]].name}</b>${best[1] ? `, ${(best[1][0] * 100).toFixed(0)}% to ${T[best[1][1]].name}` : ''}. Tokens with similar content usually win (cars with cars, clock with clock), whichever player's view they are in. Switch heads: a different slice of features gives a different pattern.`,
      `This is where cross-view consistency comes from in MIRA: a ball token in P3's view can put weight on the ball in P1's view (paper Fig. 17 shows this pattern in the trained model).`],
    [`Weighted sum: gather the values`, `Output O = A·V: each token's new colour is the weighted average of the values it attended to. Tokens that attend to similar tokens end up with "consensus" information about that object.`,
      `In MIRA every head does this with 128-number values. The 32 head outputs are concatenated (4096) and mixed by W<sub>O</sub>, then multiplied by a learned sigmoid <b>gate</b>.`],
    [`Add back onto the token (residual)`, `The token isn't replaced. The attention output is <b>added</b> to it: x ← x + gate·(O·W<sub>O</sub>). (Here W<sub>O</sub>·gate is a fixed 0.5 so you can see the blend.) The token keeps its identity and gains context.`,
      `In MIRA this is followed by the time-attention layer and then the per-token SwiGLU MLP, each also added back. That's one block; MIRA stacks 16 (§6).`],
  ][S.step];
  $('#explain').innerHTML = `<h4>${txt[0]}</h4><div>${txt[1]}</div><div class="mira"><b>In MIRA:</b> ${txt[2].replace(/^In MIRA:\s*/, '')}</div>`;
}
function render() {
  $$('#stepper button').forEach((b, i) => b.classList.toggle('on', i === S.step));
  const n = tokens().length; if (S.q >= n) S.q = n - 1;
  const R = compute(); drawPick(); drawMat(R); explain(R);
}
$$('#modeSeg button').forEach((b) => (b.onclick = () => { S.mode = b.dataset.m; S.q = S.mode === 'time' ? 7 : 0; $$('#modeSeg button').forEach((x) => x.classList.toggle('on', x === b)); render(); }));
$$('#headSeg button').forEach((b) => (b.onclick = () => { S.head = +b.dataset.h; $$('#headSeg button').forEach((x) => x.classList.toggle('on', x === b)); render(); }));
$('#scaleChk').onchange = (e) => { S.scale = e.target.checked; render(); };
render();

// ------------------------------------------------------------------ RoPE demo
{
  const c = $('#cvRope'), r = c.getContext('2d'), TH = (22 * Math.PI) / 180, qa = 0.35, ka = 1.2;
  let m = 2, n = 6;
  function draw() {
    $('#ropeM').value = m; $('#ropeN').value = n; $('#ropeMV').textContent = m; $('#ropeNV').textContent = n;
    r.fillStyle = '#0a0d13'; r.fillRect(0, 0, 420, 300);
    const cx = 210, cy = 160, R = 110;
    r.strokeStyle = '#263041'; r.beginPath(); r.arc(cx, cy, R, 0, 7); r.stroke();
    const vec = (a, col, label) => { const x = cx + R * Math.cos(a), y = cy - R * Math.sin(a); r.strokeStyle = col; r.lineWidth = 3; r.beginPath(); r.moveTo(cx, cy); r.lineTo(x, y); r.stroke(); r.fillStyle = col; r.beginPath(); r.arc(x, y, 5, 0, 7); r.fill(); r.font = '700 12px Inter, sans-serif'; r.fillText(label, x + 8, y); };
    const A = qa + m * TH, B = ka + n * TH;
    vec(qa, 'rgba(251,191,36,.25)', ''); vec(ka, 'rgba(251,146,60,.25)', '');
    vec(A, '#fbbf24', `q at pos ${m}`); vec(B, '#fb923c', `k at pos ${n}`);
    const dot = Math.cos(A - B);
    r.fillStyle = '#e6ebf2'; r.font = '700 13px ui-monospace,monospace'; r.fillText(`q·k = ${dot.toFixed(3)}`, 12, 22);
    r.fillStyle = '#93a0b4'; r.font = '12px Inter, sans-serif'; r.fillText(`offset m − n = ${m - n}`, 12, 40); r.fillText('faint arrows: before rotation', 12, 290);
    $('#ropeNote').innerHTML = `Each vector is rotated by (its position) × 22°. The score only depends on <b>m − n = ${m - n}</b>: press "shift both +1" and the arrows turn, but q·k stays <b>${dot.toFixed(3)}</b>. (Real RoPE rotates 64 such pairs per head, at many different speeds.)`;
  }
  $('#ropeM').oninput = (e) => { m = +e.target.value; draw(); };
  $('#ropeN').oninput = (e) => { n = +e.target.value; draw(); };
  $('#ropeShift').onclick = () => { if (m < 12 && n < 12) { m++; n++; } else { m -= 6; n -= 6; } draw(); };
  draw();
}

// ------------------------------------------------------------------ reshape diagram
{
  const svg = $('#rsSvg');
  function draw(mode) {
    svg.innerHTML = '';
    const F = 20, C = 16, x0 = 140, y0 = 40, w = 34, h = 13;
    svgEl('text', { x: 10, y: 22, fill: '#e6ebf2', 'font-size': 13, 'font-weight': 700 }, svg).textContent = 'context tensor: 20 (frames) × 576 (cells) × 4096 (features)  ·  drawn: 20 frames × 16 of the 576 cells';
    for (let f = 0; f < F; f++) for (let c = 0; c < C; c++) {
      let fill = '#1b2330';
      if (mode === 'space') fill = f % 2 ? 'rgba(251,191,36,.55)' : 'rgba(251,191,36,.3)';
      if (mode === 'time') fill = c % 2 ? 'rgba(251,146,60,.55)' : 'rgba(251,146,60,.3)';
      svgEl('rect', { x: x0 + f * (w + 4), y: y0 + c * (h + 2), width: w, height: h, rx: 2, fill }, svg);
    }
    svgEl('text', { x: 10, y: y0 + 8 * (h + 2), fill: '#93a0b4', 'font-size': 11.5 }, svg).textContent = 'cells ↓';
    svgEl('text', { x: x0, y: y0 + C * (h + 2) + 16, fill: '#93a0b4', 'font-size': 11.5 }, svg).textContent = 'frames →  (oldest … newest)';
    if (mode === 'space') {
      for (let f = 0; f < F; f += 1) svgEl('rect', { x: x0 + f * (w + 4) - 1.5, y: y0 - 4, width: w + 3, height: C * (h + 2) + 4, rx: 4, fill: 'none', stroke: '#fbbf24', 'stroke-width': 1 }, svg);
      $('#rsNote').innerHTML = '<b class="c-wm">Space layer</b>: each <b>column</b> (one frame, 576 tokens, all 4 views) is one attention sequence → <b>20 independent sequences of length 576</b>, no mask. Tensor viewed as (batch = 20, sequence = 576, 4096).';
    } else {
      for (let c = 0; c < C; c += 1) svgEl('rect', { x: x0 - 3, y: y0 + c * (h + 2) - 1.5, width: F * (w + 4) + 2, height: h + 3, rx: 3, fill: 'none', stroke: '#fb923c', 'stroke-width': 1 }, svg);
      svgEl('path', { d: `M${x0 + 2},${y0 + C * (h + 2) + 30} L${x0 + F * (w + 4) - 10},${y0 + C * (h + 2) + 30}`, stroke: '#fb923c', 'stroke-width': 2 }, svg);
      svgEl('path', { d: `M${x0 + F * (w + 4) - 4},${y0 + C * (h + 2) + 30} l-9,-5 v10 z`, fill: '#fb923c' }, svg);
      svgEl('text', { x: x0 + 300, y: y0 + C * (h + 2) + 46, fill: '#fb923c', 'font-size': 11.5 }, svg).textContent = 'causal: each frame looks only leftwards (earlier)';
      $('#rsNote').innerHTML = '<b class="c-noise">Time layer</b>: transpose so each <b>row</b> (one cell across 20 frames) is a sequence → <b>576 independent sequences of length 20</b>, with a causal mask. Tensor viewed as (batch = 576, sequence = 20, 4096).';
    }
  }
  $$('#rsSeg button').forEach((b) => (b.onclick = () => { $$('#rsSeg button').forEach((x) => x.classList.toggle('on', x === b)); draw(b.dataset.m); }));
  draw('space');
}

// ------------------------------------------------------------------ block diagram
{
  const svg = $('#blkSvg'), Y = 150;
  svgEl('line', { x1: 20, y1: Y, x2: 1080, y2: Y, stroke: '#a78bfa', 'stroke-width': 5, 'stroke-opacity': 0.55 }, svg);
  svgEl('text', { x: 20, y: Y + 30, fill: '#a78bfa', 'font-size': 11.5 }, svg).textContent = 'residual stream: 576 tokens × 4096 numbers per frame, flowing through';
  const sub = (x, w, col, title, lines) => {
    svgEl('rect', { x, y: 18, width: w, height: 104, rx: 10, fill: col, 'fill-opacity': 0.1, stroke: col, 'stroke-width': 1.5 }, svg);
    svgEl('text', { x: x + w / 2, y: 38, fill: '#e6ebf2', 'font-size': 12.5, 'font-weight': 700, 'text-anchor': 'middle' }, svg).textContent = title;
    lines.forEach((l, i) => (svgEl('text', { x: x + w / 2, y: 56 + i * 14, fill: '#93a0b4', 'font-size': 10.5, 'text-anchor': 'middle' }, svg).textContent = l));
    svgEl('path', { d: `M${x - 22},${Y} C ${x - 22},70 ${x - 10},70 ${x},70`, stroke: col, 'stroke-width': 1.6, fill: 'none' }, svg);
    svgEl('path', { d: `M${x + w},70 C ${x + w + 18},70 ${x + w + 26},100 ${x + w + 26},${Y - 12}`, stroke: col, 'stroke-width': 1.6, fill: 'none' }, svg);
    svgEl('circle', { cx: x + w + 26, cy: Y, r: 11, fill: '#0b0e14', stroke: col, 'stroke-width': 2 }, svg);
    svgEl('text', { x: x + w + 26, y: Y + 5, fill: col, 'font-size': 15, 'font-weight': 700, 'text-anchor': 'middle' }, svg).textContent = '+';
  };
  sub(60, 270, '#fbbf24', 'AdaLN → SPACE attention', ['Q,K,V · QK-norm · 2-D RoPE', 'softmax over 576 tokens (no mask)', '32 q-heads / 8 kv-heads · W_O · gate', '+ spatial register tokens']);
  sub(400, 270, '#fb923c', 'AdaLN → TIME attention', ['Q,K,V · QK-norm · RoPE in seconds', 'softmax over ≤20 frames (causal)', '32 q-heads / 8 kv-heads · W_O · gate', '+ temporal register tokens · KV cache']);
  sub(740, 270, '#38bdf8', 'AdaLN → SwiGLU MLP', ['per token, no mixing', '4096 → wide → 4096', 'gated activation', '(most of the block’s parameters)']);
  svgEl('rect', { x: 330, y: 245, width: 420, height: 56, rx: 10, fill: '#f472b6', 'fill-opacity': 0.1, stroke: '#f472b6' }, svg);
  svgEl('text', { x: 540, y: 268, fill: '#e6ebf2', 'font-size': 12.5, 'font-weight': 700, 'text-anchor': 'middle' }, svg).textContent = 'c = emb(4 players’ keys) + emb(noise level τ)';
  svgEl('text', { x: 540, y: 286, fill: '#93a0b4', 'font-size': 10.5, 'text-anchor': 'middle' }, svg).textContent = '→ γ, β for each AdaLN (same for every token)';
  [195, 535, 875].forEach((x) => svgEl('path', { d: `M${x},245 L${x},124`, stroke: '#f472b6', 'stroke-width': 1.2, 'stroke-dasharray': '4 3', fill: 'none' }, svg));
}
