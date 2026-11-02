import './style.css';

export const CHAPTERS = [
  { id: 'index', n: '00', title: 'The whole machine', short: 'Overview' },
  { id: 'ch1-codec', n: '01', title: 'The codec: seeing in DINO space', short: 'Codec' },
  { id: 'ch2-flow', n: '02', title: 'Flow matching & few-step sampling', short: 'Flow matching' },
  { id: 'ch3-forcing', n: '03', title: 'Diffusion forcing: why rollouts don’t drift', short: 'Diffusion forcing' },
  { id: 'ch4-transformer', n: '04', title: 'Inside the transformer', short: 'Transformer' },
  { id: 'ch5-multiplayer', n: '05', title: 'Four players, one world', short: 'Multiplayer' },
  { id: 'ch6-realtime', n: '06', title: 'Streaming at 20 fps', short: 'Real-time' },
  { id: 'ch7-evidence', n: '07', title: 'How we know it works (and where it breaks)', short: 'Evidence' },
];

export function mountNav(current) {
  if (!document.querySelector('link[rel=icon]')) { const l = document.createElement('link'); l.rel = 'icon'; l.href = 'data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22><circle cx=%228%22 cy=%228%22 r=%227%22 fill=%22%23fbbf24%22/></svg>'; document.head.appendChild(l); }
  const nav = document.createElement('nav');
  nav.className = 'topnav';
  nav.innerHTML = `<div class="inner"><a class="brand" href="./index.html">MIRA<span>/</span>explained</a>${CHAPTERS.map(
    (c) => `<a class="ch ${c.id === current ? 'active' : ''}" href="./${c.id}.html"><b>${c.n}</b>${c.short}</a>`
  ).join('')}</div>`;
  document.body.prepend(nav);
  const i = CHAPTERS.findIndex((c) => c.id === current);
  const main = document.querySelector('main');
  if (main && i >= 0) {
    const prev = CHAPTERS[i - 1], next = CHAPTERS[i + 1];
    const pager = document.createElement('div');
    pager.className = 'pager';
    pager.innerHTML = `${prev ? `<a href="./${prev.id}.html"><small>← ${prev.n}</small>${prev.title}</a>` : '<span></span>'}${next ? `<a href="./${next.id}.html" style="text-align:right"><small>${next.n} →</small>${next.title}</a>` : '<span></span>'}`;
    main.appendChild(pager);
  }
  const act = nav.querySelector('.active');
  if (act) act.scrollIntoView({ inline: 'center', block: 'nearest' });
}

// tiny helpers shared by chapters
export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
export function fitCanvas(cv, w, h) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  cv.style.width = w + 'px'; cv.style.aspectRatio = `${w} / ${h}`;
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}
// viridis-like colormap
const VIR = [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]];
export function viridis(t) {
  t = Math.max(0, Math.min(1, t)) * (VIR.length - 1);
  const i = Math.min(VIR.length - 2, Math.floor(t)), f = t - i, a = VIR[i], b = VIR[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
export function inferno(t) {
  const C = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
  t = Math.max(0, Math.min(1, t)) * (C.length - 1);
  const i = Math.min(C.length - 2, Math.floor(t)), f = t - i, a = C[i], b = C[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
