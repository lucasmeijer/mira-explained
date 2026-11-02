import { genEpisodes, ARENA } from '../src/lib/toyworld.js';
import { makeModel, Trainer, rollout, K } from '../src/lib/worldmodel.js';
import { packMLP } from '../src/lib/nn.js';
import fs from 'fs';
const mode = process.argv[2], STEPS = +(process.argv[3] || 8000);
const train = genEpisodes(600, 300, 1);
const m = makeModel(3);
const cfg = { sigmaMax: +(process.env.SMAX || 0.5), sigmaPow: +(process.env.SPOW || 1.5), residual: process.env.RESID || 'noisy' };
const tr = new Trainer(m, train, { mode, actionDropout: 0.2, ...cfg });
const curve = []; const t0 = Date.now();
for (let i = 0; i < STEPS; i += 100) {
  const lr = i < STEPS * 0.6 ? 1e-3 : i < STEPS * 0.85 ? 4e-4 : 1.5e-4;
  const l = tr.train(100, 128, lr); curve.push([i + 100, +l.toFixed(6)]);
  if (i % 1000 === 0) console.log(mode, i, l.toFixed(5), ((Date.now() - t0) / 1000).toFixed(0) + 's');
}
fs.mkdirSync('public/models/toy', { recursive: true });
const out = process.env.OUT || `public/models/toy/${mode}.json`;
fs.writeFileSync(out, JSON.stringify(packMLP(m, { mode, steps: STEPS, curve, ...cfg })));
console.log('saved');
