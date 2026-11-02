import { mountNav, $, CHAPTERS } from './nav.js';
import { svgEl } from './lib/charts.js';
mountNav('index');

const C = { enc: '#2dd4bf', lat: '#a78bfa', wm: '#fbbf24', act: '#f472b6', dec: '#38bdf8', noise: '#fb923c', mut: '#93a0b4' };
const ROLES = {
  views: { t: 'Player views', c: C.mut, role: 'What each of the 4 players sees: their own camera on the shared match. At the start of a session a short real clip is used to <b>prime</b> the model; after that every frame is generated.', kv: ['4 (players) views', '288 (px tall) × 512 (px wide) × 3 (RGB)', '20 fps'], ch: 'ch1-codec' },
  enc: { t: 'Codec encoder: frozen DINOv3-L + linear bottleneck', c: C.enc, role: 'Turns pixels into a compact, <b>meaningful</b> latent. DINOv3 (never trained by MIRA) supplies semantics; one learned linear layer merges 2 (rows) × 2 (columns) × 2 (frames) neighbours and maps 1024 → 32 (channels). Only used for priming in the live loop.', kv: ['300M frozen', '18 (rows) × 32 (cols) × 1024 (features) → 9 (rows) × 16 (cols) × 32 (channels)', '20 fps → 10 Hz', '192× fewer numbers'], breaks: 'From-scratch encoder: gFID 22.5 vs 10.7 and 1.7× more drift.', ch: 'ch1-codec' },
  win: { t: 'Context window (KV cache)', c: C.lat, role: 'The model’s entire memory: the last <b>20 latent frames (2 s)</b>, cached as keys/values so each new step costs the same. Oldest dropped, newest appended.', kv: ['20 (frames) latents', '576 tokens each = 36 (rows) × 16 (cols)', '11,520 tokens'], breaks: 'Anything older than 2 s is forgotten → replays and clocks go wrong.', ch: 'ch6-realtime' },
  wm: { t: 'World model: 5B space-time diffusion transformer', c: C.wm, role: 'Predicts the <b>velocity</b> that turns noise into the next latent for all 4 players at once. Space attention keeps the 4 views consistent; causal time attention carries motion and memory; AdaLN injects keys and noise level into every block.', kv: ['16 layers', 'width 4096', '32/8 heads (GQA)', '1–2 steps (distilled)'], breaks: 'Teacher forcing instead of diffusion forcing: gFID 32.5 vs 10.7, collapses past 4 s.', ch: 'ch4-transformer' },
  act: { t: 'Actions: 4 players × 9 keys', c: C.act, role: 'W A S D Q E Space Shift Ctrl per player. Embedded per player, concatenated in a fixed order, summed with the noise-level embedding into one vector <b>c</b> that sets every block’s scale and shift. A player can be <b>absent</b> → the model plays them (autopilot).', kv: ['9 keys', 'per-player embedding', 'AdaLN', 'dropout → autopilot'], breaks: 'Single-player conditioning: cars get lost or merge into the ball.', ch: 'ch5-multiplayer' },
  noise: { t: 'Noise z₀ & flow matching', c: C.noise, role: 'Every new latent starts as Gaussian noise and is pushed toward a plausible future along the learned flow. Few-step distillation lets 1–2 big steps replace ~10 small ones.', kv: ['z_τ = τ z₁ + (1−τ) z₀', 'v ≈ z₁ − z₀', 'PSD distillation'], ch: 'ch2-flow' },
  new: { t: 'New latent frame', c: C.lat, role: 'The model’s guess for the next 100 ms of all four views: 4 (players) × 9 (rows) × 16 (columns) × 32 (channels) = 18,432 numbers. It is decoded, <b>and</b> appended to the context window (roll forward), so the model keeps building on its own output.', kv: ['4 (players) × 9 (rows) × 16 (columns) × 32 (channels) = 18,432 numbers', '= 100 ms of game'], breaks: 'Without diffusion forcing, errors in these self-generated latents compound into drift.', ch: 'ch3-forcing' },
  dec: { t: 'Codec decoder: causal space-time ViT', c: C.dec, role: 'Paints pixels back: upsample 9 (rows) × 16 (cols) → 18 (rows) × 32 (cols) → attention (full in space, causal in time, looking at the last 3 latents) → 2 frames per latent. Trained with L1 + LPIPS + DINO-feature loss, no GAN.', kv: ['0.45B', '9 (rows) × 16 (cols) → 18 (rows) × 32 (cols) tokens', '1 latent → 2 frames'], breaks: 'Smaller (Base) decoder: blurrier reconstructions.', ch: 'ch1-codec' },
  out: { t: 'Displayed frames', c: C.mut, role: 'Two frames per player per step, queued and sent at a steady 50 ms cadence. One full step takes ~70 ms on a B200 → ~35 ms per frame, inside the 50 ms budget.', kv: ['20 fps', '~70 ms / step', '1× B200'], ch: 'ch6-realtime' },
};

const svg = $('#mapSvg');
const nodes = {};
function node(id, x, y, w, h, title, sub, extra) {
  const g = svgEl('g', { class: 'node' }, svg); const c = ROLES[id].c;
  svgEl('rect', { x, y, width: w, height: h, rx: 12, fill: c, 'fill-opacity': 0.12, stroke: c, 'stroke-width': 1.6 }, g);
  svgEl('text', { x: x + w / 2, y: y + 22, fill: '#e6ebf2', 'font-size': 13.5, 'font-weight': 700, 'text-anchor': 'middle' }, g).textContent = title;
  sub.split('\n').forEach((s, i) => (svgEl('text', { x: x + w / 2, y: y + 40 + i * 14, fill: '#93a0b4', 'font-size': 11, 'text-anchor': 'middle' }, g).textContent = s));
  if (extra) extra(g);
  g.addEventListener('click', () => select(id));
  nodes[id] = { g, x, y, w, h };
  return g;
}
function arrow(x1, y1, x2, y2, color = '#5b6b85', id, dash) {
  const p = svgEl('path', { d: `M${x1},${y1} L${x2 - 7},${y2}`, stroke: color, 'stroke-width': 2, fill: 'none', 'stroke-dasharray': dash || '' }, svg);
  const a = Math.atan2(y2 - y1, x2 - x1);
  svgEl('path', { d: `M${x2},${y2} L${x2 - 10 * Math.cos(a - 0.4)},${y2 - 10 * Math.sin(a - 0.4)} L${x2 - 10 * Math.cos(a + 0.4)},${y2 - 10 * Math.sin(a + 0.4)} Z`, fill: color }, svg);
  if (id) arrows[id] = p;
  return p;
}
const arrows = {};
// views (input)
node('views', 14, 130, 150, 190, '4 player views', '288 (px tall) × 512 (px wide)', (g) => {
  svgEl('image', { href: '/frames/hero/f_0100.jpg', x: 24, y: 185, width: 130, height: 73, preserveAspectRatio: 'none' }, g);
  svgEl('text', { x: 89, y: 280, fill: '#4ade80', 'font-size': 11, 'text-anchor': 'middle' }, g).textContent = 'real clip primes';
  svgEl('text', { x: 89, y: 294, fill: '#4ade80', 'font-size': 11, 'text-anchor': 'middle' }, g).textContent = 'the session';
});
node('enc', 196, 140, 160, 170, 'Codec encoder', 'DINOv3-L (frozen)\n+ linear bottleneck\n18 (rows) × 32 (cols)\n× 1024 (features)\n→ 9 (rows) × 16 (cols)\n× 32 (channels) @10Hz');
node('win', 390, 120, 150, 210, 'Context window', 'last 20 latents', (g) => {
  for (let k = 0; k < 20; k++) { const xx = 404 + (k % 10) * 12.5, yy = 170 + Math.floor(k / 10) * 60; svgEl('rect', { class: 'slot', x: xx, y: yy, width: 10, height: 50, rx: 2, fill: k < 12 ? '#4ade80' : C.wm, 'fill-opacity': 0.55 }, g); }
  svgEl('text', { x: 465, y: 310, fill: '#93a0b4', 'font-size': 11, 'text-anchor': 'middle' }, g).textContent = 'KV cache · 2 s';
});
node('wm', 590, 110, 210, 230, 'World model', '5B diffusion transformer', (g) => {
  const rows = [['space attention', 'all 4 views'], ['time attention', 'causal'], ['AdaLN · SwiGLU', 'keys + τ']];
  rows.forEach(([a, b], i) => { svgEl('rect', { x: 606, y: 168 + i * 46, width: 178, height: 38, rx: 7, fill: '#0b0e14', stroke: C.wm, 'stroke-opacity': 0.5 }, g); svgEl('text', { x: 695, y: 184 + i * 46, fill: '#e6ebf2', 'font-size': 11.5, 'text-anchor': 'middle', 'font-weight': 600 }, g).textContent = a; svgEl('text', { x: 695, y: 198 + i * 46, fill: '#93a0b4', 'font-size': 10.5, 'text-anchor': 'middle' }, g).textContent = b; });
  svgEl('text', { x: 695, y: 320, fill: C.wm, 'font-size': 11, 'text-anchor': 'middle' }, g).textContent = '× 16 blocks · 1–2 passes';
});
node('act', 590, 8, 210, 66, 'Actions', '4 players × 9 keys\n→ per-player emb → AdaLN', null);
node('noise', 610, 370, 170, 52, 'Noise z₀', 'flow matching · few steps', null);
node('new', 826, 160, 112, 130, 'New latent', '4 (players) ×\n9 (rows) ×\n16 (columns) ×\n32 (channels)\n= next 100 ms');
node('dec', 956, 140, 150, 110, 'Codec decoder', 'causal ViT · 0.45B\n1 latent → 2 frames');
node('out', 956, 280, 150, 100, 'Display', '2 frames / view\n20 fps · queue');
arrow(164, 225, 196, 225, '#5b6b85', 'v2e');
arrow(356, 225, 390, 225, '#5b6b85', 'e2w');
arrow(540, 225, 590, 225, '#5b6b85', 'w2m');
arrow(695, 74, 695, 110, C.act, 'a2m');
arrow(695, 370, 695, 340, C.noise, 'n2m');
arrow(800, 225, 826, 225, '#5b6b85', 'm2n');
arrow(938, 205, 956, 195, '#5b6b85', 'n2d');
arrow(1031, 250, 1031, 280, '#5b6b85', 'd2o');
// roll-forward loop
svgEl('path', { id: 'loopPath', d: 'M882,290 L882,452 L465,452 L465,334', stroke: C.lat, 'stroke-width': 2, fill: 'none', 'stroke-dasharray': '6 5' }, svg);
svgEl('path', { d: 'M465,330 l-6,12 l12,0 z', fill: C.lat }, svg);
svgEl('text', { x: 480, y: 470, fill: C.lat, 'font-size': 11.5 }, svg).textContent = 'roll forward: append the newest latent to the context, drop the oldest';
svgEl('text', { x: 196, y: 330, fill: '#66728a', 'font-size': 10.5 }, svg).textContent = 'encoder only runs to prime';
const pulse = svgEl('circle', { r: 7, fill: '#fff', opacity: 0 }, svg);
const tickLbl = svgEl('text', { x: 14, y: 30, fill: '#e6ebf2', 'font-size': 13, 'font-weight': 700 }, svg);
const tickSub = svgEl('text', { x: 14, y: 48, fill: '#93a0b4', 'font-size': 11.5 }, svg);
const tickSub2 = svgEl('text', { x: 14, y: 64, fill: '#93a0b4', 'font-size': 11.5 }, svg);

function select(id) {
  const r = ROLES[id];
  Object.entries(nodes).forEach(([k, n]) => n.g.querySelector('rect').setAttribute('stroke-width', k === id ? 3.5 : 1.6));
  const ch = CHAPTERS.find((c) => c.id === r.ch);
  $('#roleCard').innerHTML = `<h3 style="color:${r.c}">${r.t}</h3><div>${r.role}</div><div class="kv">${r.kv.map((k) => `<span>${k}</span>`).join('')}</div>${r.breaks ? `<div class="muted" style="font-size:13.5px"><b style="color:var(--bad)">Without it:</b> ${r.breaks}</div>` : ''}<div style="margin-top:8px"><a href="./${ch.id}.html">→ Chapter ${ch.n}: ${ch.title}</a></div>`;
}
select('wm');

// one tick animation
function moveAlong(path, ms) {
  return new Promise((res) => {
    const L = path.getTotalLength(), t0 = performance.now(); pulse.setAttribute('opacity', 1);
    const f = () => { const t = Math.min(1, (performance.now() - t0) / ms), p = path.getPointAtLength(L * t); pulse.setAttribute('cx', p.x); pulse.setAttribute('cy', p.y); if (t < 1) requestAnimationFrame(f); else res(); };
    f();
  });
}
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
let busy = false;
$('#tickBtn').onclick = async () => {
  if (busy) return; busy = true;
  const say = (a, b = '', c = '') => { tickLbl.textContent = a; tickSub.textContent = b; tickSub2.textContent = c; };
  say('t = 0 ms', 'read all 4 players’ keys'); select('act'); pulse.setAttribute('fill', C.act); await moveAlong(arrows.a2m, 600);
  say('world model pass 1 of 2', 'noise latent + 20 cached latents + keys', 'space attn ↔ time attn ↔ AdaLN, ×16'); select('noise'); pulse.setAttribute('fill', C.noise); await moveAlong(arrows.n2m, 500);
  select('wm'); pulse.setAttribute('opacity', 0);
  const r = nodes.wm.g.querySelector('rect');
  for (let k = 0; k < 2; k++) { say(`world model pass ${k + 1} of 2`, k ? 'second (final) denoising step' : 'big step from noise (distilled sampler)', 'space attn ↔ time attn ↔ AdaLN, ×16'); r.setAttribute('fill-opacity', 0.45); await wait(380); r.setAttribute('fill-opacity', 0.12); await wait(200); }
  say('new latent ready', '4 (players) × 9 (rows) × 16 (cols) × 32 (channels) = next 100 ms'); select('new'); pulse.setAttribute('fill', C.lat); await moveAlong(arrows.m2n, 400);
  say('decode → 2 frames per view', 'causal ViT, looks at last 3 latents'); select('dec'); pulse.setAttribute('fill', C.dec); await moveAlong(arrows.n2d, 400); await wait(300);
  say('≈ 70 ms: 2 frames → queue → screen', 'shown 50 ms apart → 20 fps'); select('out'); await moveAlong(arrows.d2o, 400);
  say('roll forward', 'append the new latent to the cache, drop the oldest', 'next tick conditions on the model’s own output'); select('win'); pulse.setAttribute('fill', C.lat); await moveAlong($('#loopPath'), 900);
  // shift window colours
  const slots = [...svg.querySelectorAll('.slot')]; const nGold = slots.filter((s) => s.getAttribute('fill') === C.wm).length;
  slots.forEach((s, i) => s.setAttribute('fill', i >= 19 - nGold ? C.wm : '#4ade80'));
  pulse.setAttribute('opacity', 0); say('', ''); busy = false;
};

// chapter cards
const DESC = {
  'ch1-codec': ['Run the real DINOv3 encoder on four player views, click any object to find it across views, and squeeze the features through the 32-channel bottleneck.', ['real', 'paper']],
  'ch2-flow': ['Train a flow-matching model live, watch 1-step sampling fail, then apply MIRA’s self-distillation and watch it succeed in one step.', ['real']],
  'ch3-forcing': ['Two learned world models, identical except for teacher vs. diffusion forcing, play the same match beside the true engine. One drifts away.', ['toy', 'viz', 'paper']],
  'ch4-transformer': ['Built up from makemore: tokens in a 36 (rows) × 16 (columns) × 20 (frames) grid, attention worked out by hand, space vs. time attention, and how a keypress reaches every token through AdaLN.', ['real', 'viz', 'paper']],
  'ch5-multiplayer': ['Play inside a learned world model, hand players to autopilot (action dropout), and probe whether the model knows whose key moves which car.', ['toy', 'real']],
  'ch6-realtime': ['The rolling KV-cache window and a working simulator of the 20 fps serving loop. Break distillation or the cache and watch the stutter.', ['viz', 'toy']],
  'ch7-evidence': ['FID, ARR, physics probes and human Elo. Run a mini ARR study, then trace every failure mode back to the component that causes it.', ['real', 'paper']],
};
$('#chapters').innerHTML = CHAPTERS.filter((c) => c.id !== 'index').map((c) => `<a href="./${c.id}.html"><div class="n">CHAPTER ${c.n}</div><h3>${c.title}</h3><p>${DESC[c.id][0]}</p><div class="badges">${DESC[c.id][1].map((b) => `<span class="badge ${b}">${b}</span>`).join('')}</div></a>`).join('');
