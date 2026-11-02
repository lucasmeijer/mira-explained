// Main-thread client for the DINOv3 worker.
export class Dino {
  constructor() {
    this.w = new Worker(new URL('../dino.worker.js', import.meta.url), { type: 'module' });
    this.pending = new Map(); this.nextId = 1; this.onprogress = null;
    this.w.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') { this.onprogress && this.onprogress(m.loaded / m.total); return; }
      const p = this.pending.get(m.id); if (!p) return; this.pending.delete(m.id);
      if (m.type === 'error') p.reject(new Error(m.message)); else p.resolve(m);
    };
    this.queue = Promise.resolve();
  }
  call(msg, transfer) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => { this.pending.set(id, { resolve, reject }); this.w.postMessage({ ...msg, id }, transfer || []); });
  }
  init(variant = 'fp32', pref = 'auto') { if (!this.ready) this.ready = this.call({ type: 'init', variant, pref }).then((m) => (this.provider = m.provider)); return this.ready; }
  embed(rgba, W, H) {
    // serialise requests
    const job = this.queue.then(() => this.call({ type: 'embed', rgba, W, H }));
    this.queue = job.catch(() => {});
    return job;
  }
}

let shared;
export function getDino() { if (!shared) shared = new Dino(); return shared; }

export function loadImage(src) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; });
}

// Crop a region of an image into a W×H RGBA buffer (rescaled).
export function cropRGBA(img, sx, sy, sw, sh, W = 512, H = 288) {
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const g = c.getContext('2d'); g.drawImage(img, sx, sy, sw, sh, 0, 0, W, H);
  return { rgba: g.getImageData(0, 0, W, H).data, canvas: c };
}

// Frame sources: each yields 4 synchronised player views.
export const SOURCES = {
  mira: {
    label: 'MIRA-generated match (4 views)',
    frames: ['f_0020', 'f_0060', 'f_0100', 'f_0140', 'f_0180', 'f_0220', 'f_0270', 'f_0320'].map((f) => `/frames/hero/${f}.jpg`),
    views: [[0, 0, 512, 288], [512, 0, 512, 288], [0, 288, 512, 288], [512, 288, 512, 288]],
  },
  real: {
    label: 'Real recording from the dataset (4 views, with overlays)',
    frames: ['r_1_0', 'r_2_0', 'r_4_5', 'r_7_0'].map((f) => `/frames/real/${f}.jpg`),
    views: [[6, 6, 623, 349], [637, 6, 623, 349], [6, 364, 623, 349], [637, 364, 623, 349]],
  },
};
