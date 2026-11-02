import { mountNav, $, $$ } from './nav.js';
import { svgEl, lineChart } from './lib/charts.js';
import { loadImage } from './lib/dino.js';
import { MLP, unpackMLP, mulberry32 } from './lib/nn.js';
import { genEpisodes, ARENA } from './lib/toyworld.js';
import { rollout, K } from './lib/worldmodel.js';
import { arenaMapper } from './lib/toydraw.js';
mountNav('ch8-training');

// ---------------------------------------------------------------- data-parallel schematic
{
  const el = $('#gpus');
  for (let i = 0; i < 32; i++) el.insertAdjacentHTML('beforeend', `<div>${i}</div>`);
  const cells = [...el.children];
  const set = (bg, bd) => cells.forEach((c) => { c.style.background = bg; c.style.borderColor = bd; });
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  $('#gpuStep').onclick = async () => {
    $('#gpuStep').disabled = true;
    $('#gpuNote').innerHTML = '① each GPU gets <b>1 different clip</b> (32 clips = the global batch)…'; set('rgba(56,189,248,.18)', 'rgba(56,189,248,.6)'); await wait(1100);
    $('#gpuNote').innerHTML = '② …runs forward + backward on its own clip → its own gradient…'; set('rgba(251,146,60,.2)', 'rgba(251,146,60,.6)'); await wait(1100);
    $('#gpuNote').innerHTML = '③ …all 32 gradients are <b>averaged</b> over the fast interconnect (all-reduce)…'; set('rgba(167,139,250,.25)', 'rgba(167,139,250,.8)'); await wait(1100);
    $('#gpuNote').innerHTML = '④ …and every GPU applies the <b>same</b> AdamW update, so the 32 copies stay identical. Repeat 100,000×.'; set('rgba(74,222,128,.18)', 'rgba(74,222,128,.6)'); await wait(1300);
    set('', ''); $('#gpuStep').disabled = false;
  };
}

// ---------------------------------------------------------------- pipeline diagrams with gradient flow
function pipeline(svg, nodes, edges, lossX, lossY) {
  const N = {};
  nodes.forEach((n) => {
    const g = svgEl('g', {}, svg);
    svgEl('rect', { x: n.x, y: n.y, width: n.w, height: n.h, rx: 10, fill: n.c, 'fill-opacity': n.frozen ? 0.05 : 0.14, stroke: n.c, 'stroke-width': 1.6, 'stroke-dasharray': n.frozen ? '5 4' : '' }, g);
    svgEl('text', { x: n.x + n.w / 2, y: n.y + 20, fill: '#e6ebf2', 'font-size': 12.5, 'font-weight': 700, 'text-anchor': 'middle' }, g).textContent = n.t;
    (n.s || '').split('\n').forEach((l, i) => (svgEl('text', { x: n.x + n.w / 2, y: n.y + 37 + i * 13, fill: '#93a0b4', 'font-size': 10.5, 'text-anchor': 'middle' }, g).textContent = l));
    if (n.tag) svgEl('text', { x: n.x + n.w / 2, y: n.y + n.h + 15, fill: n.frozen ? '#66728a' : '#4ade80', 'font-size': 10.5, 'font-weight': 700, 'text-anchor': 'middle' }, g).textContent = n.tag;
    N[n.id] = { ...n, g };
  });
  const paths = edges.map(([a, b, opt = {}]) => {
    const A = N[a], B = N[b];
    const x1 = opt.x1 ?? A.x + A.w, y1 = opt.y1 ?? A.y + A.h / 2, x2 = opt.x2 ?? B.x, y2 = opt.y2 ?? B.y + B.h / 2;
    const p = svgEl('path', { d: `M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}`, stroke: opt.c || '#5b6b85', 'stroke-width': 1.8, fill: 'none', 'stroke-dasharray': opt.dash || '' }, svg);
    return { a, b, p };
  });
  const dot = svgEl('circle', { r: 6, opacity: 0 }, svg);
  const anim = (path, ms, rev, color) => new Promise((res) => {
    const L = path.getTotalLength(), t0 = performance.now(); dot.setAttribute('fill', color); dot.setAttribute('opacity', 1);
    const f = () => { const t = Math.min(1, (performance.now() - t0) / ms), q = path.getPointAtLength(L * (rev ? 1 - t : t)); dot.setAttribute('cx', q.x); dot.setAttribute('cy', q.y); t < 1 ? requestAnimationFrame(f) : res(); };
    f();
  });
  const flash = (id, color) => { const r = N[id].g.querySelector('rect'); const old = r.getAttribute('fill-opacity'); r.setAttribute('fill', color); r.setAttribute('fill-opacity', 0.45); setTimeout(() => { r.setAttribute('fill', N[id].c); r.setAttribute('fill-opacity', old); }, 450); };
  return {
    async forward(order) { for (const [a, b] of order) { const e = paths.find((p) => p.a === a && p.b === b); flash(a, '#38bdf8'); await anim(e.p, 450, false, '#38bdf8'); } dot.setAttribute('opacity', 0); },
    async backward(order) {
      for (const [a, b] of order) {
        const e = paths.find((p) => p.a === a && p.b === b);
        if (N[a].frozen) { flash(a, '#66728a'); continue; }
        await anim(e.p, 450, true, '#f87171'); flash(a, N[a].frozen ? '#66728a' : '#f87171');
      }
      dot.setAttribute('opacity', 0);
    },
  };
}
const NOTE = (svg, x, y, t, c = '#93a0b4') => (svgEl('text', { x, y, fill: c, 'font-size': 11.5 }, svg).textContent = t);

{
  const svg = $('#svgCodec');
  const P = pipeline(svg, [
    { id: 'clip', x: 10, y: 80, w: 130, h: 70, c: '#93a0b4', t: 'video clip x', s: '40 frames of\none player', tag: 'data' },
    { id: 'dino', x: 175, y: 70, w: 150, h: 90, c: '#2dd4bf', t: 'DINOv3-L', s: 'per-frame features\n18 (rows) × 32 (cols)\n× 1024', tag: 'FROZEN · no gradients', frozen: true },
    { id: 'bott', x: 360, y: 75, w: 140, h: 80, c: '#a78bfa', t: 'bottleneck', s: 'linear 2×2×2\n1024 → 32 (channels)', tag: 'TRAINED' },
    { id: 'dec', x: 535, y: 70, w: 150, h: 90, c: '#38bdf8', t: 'decoder', s: 'space-time ViT\n0.45B params', tag: 'TRAINED' },
    { id: 'rec', x: 720, y: 80, w: 120, h: 70, c: '#93a0b4', t: 'reconstruction x̂', s: '40 frames' },
    { id: 'loss', x: 885, y: 55, w: 200, h: 120, c: '#f87171', t: 'loss', s: 'L1(x, x̂)\n+ λp · LPIPS(x, x̂)\n+ λd · ‖DINO(x) − DINO(x̂)‖²\nλ’s auto-balanced', tag: 'compares to the original x' },
  ], [['clip', 'dino'], ['dino', 'bott'], ['bott', 'dec'], ['dec', 'rec'], ['rec', 'loss'], ['clip', 'loss', { x1: 75, y1: 80, x2: 985, y2: 55, dash: '4 4', c: '#66728a' }]]);
  NOTE(svg, 10, 30, 'Forward: clip → frozen DINO → bottleneck → decoder → reconstruction → compare with the original clip.');
  NOTE(svg, 10, 48, 'Backward: gradients flow back into the decoder and bottleneck, and stop at DINOv3 (frozen). No world model anywhere.', '#f87171');
  NOTE(svg, 10, 235, 'Data: single-player 4-s chunks (288×512, 20 fps) from the 10k-hour corpus · global batch 32 clips · AdamW, lr 2·10⁻⁴ · 400k steps on 32× B200 (live codec)');
  $('#cFwd').onclick = () => P.forward([['clip', 'dino'], ['dino', 'bott'], ['bott', 'dec'], ['dec', 'rec'], ['rec', 'loss']]);
  $('#cBwd').onclick = () => P.backward([['rec', 'loss'], ['dec', 'rec'], ['bott', 'dec'], ['dino', 'bott']]);
}
{
  const svg = $('#svgWM');
  const P = pipeline(svg, [
    { id: 'clip', x: 10, y: 70, w: 120, h: 80, c: '#93a0b4', t: '4-s clip', s: '4 players ×\n80 frames\n+ keys', tag: 'data' },
    { id: 'enc', x: 160, y: 70, w: 140, h: 80, c: '#2dd4bf', t: 'codec encoder', s: 'DINOv3 + bottleneck\n(from stage 1)', tag: 'FROZEN', frozen: true },
    { id: 'z1', x: 330, y: 65, w: 150, h: 90, c: '#a78bfa', t: 'clean latents z₁', s: '40 (frames) × 36 (rows)\n× 16 (cols)\n× 32 (channels)' },
    { id: 'mix', x: 510, y: 65, w: 150, h: 90, c: '#fb923c', t: 'noise per frame', s: 'τ ~ U(0,1) per frame\nz_τ = τ·z₁ + (1−τ)·z₀' },
    { id: 'wm', x: 690, y: 55, w: 160, h: 110, c: '#fbbf24', t: 'world model', s: '5B transformer\ncausal in time\n+ keys via AdaLN\n(action dropout)', tag: 'TRAINED' },
    { id: 'loss', x: 880, y: 65, w: 205, h: 90, c: '#f87171', t: 'loss', s: '‖ v_pred − (z₁ − z₀) ‖²\nover all 40 frames\n(+ self-distillation term)' },
    { id: 'keys', x: 510, y: 200, w: 150, h: 60, c: '#f472b6', t: 'keys of P1–P4', s: 'per-player embedding' },
  ], [['clip', 'enc'], ['enc', 'z1'], ['z1', 'mix'], ['mix', 'wm'], ['wm', 'loss'], ['keys', 'wm', { x1: 660, y1: 230, x2: 770, y2: 165 }], ['z1', 'loss', { x1: 405, y1: 65, x2: 985, y2: 65, dash: '4 4', c: '#66728a' }], ['clip', 'keys', { x1: 70, y1: 150, x2: 510, y2: 230, dash: '4 4', c: '#66728a' }]]);
  NOTE(svg, 10, 25, 'Forward: encode with the frozen codec → noise each frame independently → predict velocity → compare with the true direction z₁ − z₀.');
  NOTE(svg, 10, 43, 'Backward: only the world model’s weights change. The decoder is not used at all.', '#f87171');
  NOTE(svg, 700, 255, 'Stage 2: 1 view, 1 key stream, 30k steps');
  NOTE(svg, 700, 272, 'Stage 3: 4 tiled views, 4 key streams, 100k steps');
  NOTE(svg, 700, 289, 'AdamW lr 1·10⁻⁴ · weight decay 0.1 · EMA of weights');
  $('#wFwd').onclick = () => P.forward([['clip', 'enc'], ['enc', 'z1'], ['z1', 'mix'], ['mix', 'wm'], ['wm', 'loss']]);
  $('#wBwd').onclick = () => P.backward([['wm', 'loss'], ['mix', 'wm'], ['enc', 'z1']]);
}

// ---------------------------------------------------------------- decoder training demo
{
  const LAT = await (await fetch('/models/toy/latents.json')).json();
  const b64 = (s) => { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Float32Array(u.buffer); };
  const imgCache = new Map();
  const views = [];
  for (const v of LAT.views) {
    if (!imgCache.has(v.src)) imgCache.set(v.src, await loadImage(v.src));
    const c = document.createElement('canvas'); c.width = 256; c.height = 144;
    const g = c.getContext('2d'); g.drawImage(imgCache.get(v.src), ...v.box, 0, 0, 256, 144);
    const px = g.getImageData(0, 0, 256, 144).data;
    views.push({ set: v.set, z: b64(v.z), px, canvas: c });
  }
  // normalise latents with training statistics
  const train = views.filter((v) => v.set === 'train');
  const mu = new Float32Array(32), sd = new Float32Array(32);
  train.forEach((v) => { for (let i = 0; i < 144; i++) for (let c = 0; c < 32; c++) mu[c] += v.z[i * 32 + c] / (train.length * 144); });
  train.forEach((v) => { for (let i = 0; i < 144; i++) for (let c = 0; c < 32; c++) sd[c] += (v.z[i * 32 + c] - mu[c]) ** 2 / (train.length * 144); });
  for (let c = 0; c < 32; c++) sd[c] = Math.sqrt(sd[c]) || 1;
  views.forEach((v) => { v.zn = new Float32Array(144 * 32); for (let i = 0; i < 144; i++) for (let c = 0; c < 32; c++) v.zn[i * 32 + c] = (v.z[i * 32 + c] - mu[c]) / sd[c]; });
  const IN = 9 * 32, OUT = 16 * 16 * 3;
  function input(X, o, v, r, c) {
    let p = o;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      const rr = r + dr, cc = c + dc, ok = rr >= 0 && rr < 9 && cc >= 0 && cc < 16;
      for (let k = 0; k < 32; k++) X[p++] = ok ? v.zn[(rr * 16 + cc) * 32 + k] : 0;
    }
  }
  function target(Y, o, v, r, c) {
    let p = o;
    for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { const q = ((r * 16 + y) * 256 + c * 16 + x) * 4; for (let k = 0; k < 3; k++) Y[p++] = v.px[q + k] / 127.5 - 1; }
  }
  let net, rng, steps, lossPts, running = false;
  const reset = () => { net = new MLP([IN, 256, OUT], 7); rng = mulberry32(1); steps = 0; lossPts = []; running = false; $('#decTrain').textContent = '▶ Train'; renderAll(); };
  const pick = { train: 0, heldout: 0, desert: 0 };
  const sets = { train: views.filter((v) => v.set === 'train'), heldout: views.filter((v) => v.set === 'heldout'), desert: views.filter((v) => v.set === 'desert') };
  function decode(v) {
    const X = new Float32Array(144 * IN); for (let r = 0; r < 9; r++) for (let c = 0; c < 16; c++) input(X, (r * 16 + c) * IN, v, r, c);
    return net.forward(X, 144);
  }
  function err(v, Yp) { let e = 0; for (let r = 0; r < 9; r++) for (let c = 0; c < 16; c++) for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) { const q = ((r * 16 + y) * 256 + c * 16 + x) * 4, o = (r * 16 + c) * OUT + (y * 16 + x) * 3; for (let k = 0; k < 3; k++) e += Math.abs(Math.max(-1, Math.min(1, Yp[o + k])) - (v.px[q + k] / 127.5 - 1)); } return e / (144 * OUT) * 127.5; }
  function paint(cv, v, Yp) {
    const g = cv.getContext('2d');
    if (!Yp) { g.drawImage(v.canvas, 0, 0); return; }
    const id = g.createImageData(256, 144);
    for (let r = 0; r < 9; r++) for (let c = 0; c < 16; c++) for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
      const q = ((r * 16 + y) * 256 + c * 16 + x) * 4, o = (r * 16 + c) * OUT + (y * 16 + x) * 3;
      for (let k = 0; k < 3; k++) id.data[q + k] = Math.max(0, Math.min(255, (Yp[o + k] + 1) * 127.5)); id.data[q + 3] = 255;
    }
    g.putImageData(id, 0, 0);
  }
  function renderAll() {
    const show = [['train', 'dA', 'eA'], ['heldout', 'dB', 'eB'], ['desert', 'dC', 'eC']];
    show.forEach(([s, cv, e]) => {
      const v = sets[s][pick[s] % sets[s].length];
      paint($('#' + cv + '0'), v); const Yp = decode(v); paint($('#' + cv + '1'), v, steps ? Yp : null);
      if (!steps) { $('#' + cv + '1').getContext('2d').fillStyle = '#0a0d13'; $('#' + cv + '1').getContext('2d').fillRect(0, 0, 256, 144); }
      // average error over the whole set
      const sub = sets[s].filter((_, i) => i % Math.ceil(sets[s].length / 3) === 0); const avg = sub.reduce((a, vv) => a + err(vv, decode(vv)), 0) / sub.length;
      $('#' + e).textContent = steps ? `${avg.toFixed(1)} / 255 per pixel` : '–';
      $('#' + e).style.color = s === 'desert' ? 'var(--bad)' : s === 'heldout' ? 'var(--wm)' : 'var(--good)';
      if (steps) lossPts[s] = lossPts[s] || [], lossPts[s].push([steps, avg]);
    });
    if (steps) lineChart($('#decLoss'), {
      title: 'mean pixel error (L1, 0–255 scale)', xlabel: 'training step', ymin: 0, height: 210, width: 560,
      series: [{ name: 'training frames', color: '#4ade80', points: lossPts.train || [] }, { name: 'unseen, same arena', color: '#fbbf24', points: lossPts.heldout || [] }, { name: 'unseen arena', color: '#f87171', points: lossPts.desert || [] }],
    });
  }
  let lastR = 0;
  const loop2 = () => { const t0 = performance.now(); while (performance.now() - t0 < 40) { const B = 48, X = new Float32Array(B * IN), Y = new Float32Array(B * OUT); for (let b = 0; b < B; b++) { const v = train[Math.floor(rng() * train.length)], r = Math.floor(rng() * 9), c = Math.floor(rng() * 16); input(X, b * IN, v, r, c); target(Y, b * OUT, v, r, c); } net.trainMSE(X, Y, B, steps < 1500 ? 1e-3 : 4e-4); steps++; } $('#decInfo').textContent = `step ${steps}`; if (performance.now() - lastR > 2500 || !running) { renderAll(); lastR = performance.now(); } if (running) requestAnimationFrame(loop2); };
  $('#decTrain').onclick = () => { running = !running; $('#decTrain').textContent = running ? '❚❚ Pause' : '▶ Train'; if (running) loop2(); else renderAll(); };
  $('#decReset').onclick = reset;
  $('#decNext').onclick = () => { pick.train++; pick.heldout++; pick.desert++; renderAll(); };
  reset();
}

// ---------------------------------------------------------------- world model training in miniature
{
  const cv = $('#wmCv'), g = cv.getContext('2d'), M = arenaMapper(520, 330);
  const ep = genEpisodes(1, 120, 2024)[0];
  const H = 90, truth = ep.S.slice(K - 1, K - 1 + H + 1), acts = ep.A.slice(K - 1, K - 1 + H);
  let pred = null, step = 0;
  const lossPts = [];
  function draw() {
    g.fillStyle = '#0a0d13'; g.fillRect(0, 0, 520, 330);
    g.fillStyle = '#0f1a14'; g.fillRect(M.X(-ARENA.w), M.Y(ARENA.h), 2 * ARENA.w * M.s, 2 * ARENA.h * M.s);
    g.strokeStyle = 'rgba(255,255,255,.18)'; g.strokeRect(M.X(-ARENA.w), M.Y(ARENA.h), 2 * ARENA.w * M.s, 2 * ARENA.h * M.s);
    const path = (S, o, col, w) => { g.strokeStyle = col; g.lineWidth = w; g.beginPath(); S.forEach((s, i) => { const x = Math.max(-50, Math.min(570, M.X(s[o]))), y = Math.max(-50, Math.min(380, M.Y(s[o + 1]))); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); };
    path(truth, 8, '#e6ebf2', 2.5); path(truth, 0, 'rgba(56,189,248,.6)', 1.5); path(truth, 4, 'rgba(251,146,60,.6)', 1.5);
    if (pred) { path(pred, 8, '#fbbf24', 2.5); }
    g.fillStyle = '#e6ebf2'; g.font = '700 12px Inter, sans-serif'; g.fillText(`6-s rollout · weights after ${step} training steps`, 10, 18);
    g.font = '11px Inter, sans-serif'; g.fillStyle = '#93a0b4'; g.fillText('white = real ball path · amber = model’s ball path · blue/orange = real car paths', 10, 322);
  }
  draw();
  $('#wmTrain').onclick = () => {
    $('#wmTrain').disabled = true; lossPts.length = 0;
    const w = new Worker(new URL('./toytrain.worker.js', import.meta.url), { type: 'module' });
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') { lossPts.push([m.step, m.loss]); $('#wmInfo').textContent = `step ${m.step} / ${m.steps} · loss ${m.loss.toFixed(4)}`; if (m.step % 200 === 0) lineChart($('#wmLoss'), { title: 'training loss (MSE)', series: [{ name: 'loss', color: '#fbbf24', points: lossPts }], xlabel: 'training step', logy: true, height: 230, width: 520, legend: false }); }
      if (m.type === 'snap') { const net = unpackMLP(m.model); step = m.step; pred = [truth[0], ...rollout(net, ep.S.slice(0, K), acts, 0, 1)]; draw(); }
      if (m.type === 'done') { w.terminate(); $('#wmTrain').disabled = false; $('#wmTrain').textContent = '↻ Train again'; $('#wmInfo').textContent += ' · done'; }
    };
    w.postMessage({ mode: 'diffusion', steps: 3000, episodes: 300, seed: Math.floor(Math.random() * 1000), snapshotEvery: 250 });
  };
}
