// Trains a Rocket Pong world model off the main thread (same code that produced the shipped weights).
import { genEpisodes } from './lib/toyworld.js';
import { makeModel, Trainer } from './lib/worldmodel.js';
import { packMLP } from './lib/nn.js';

self.onmessage = (e) => {
  const { mode, steps = 3000, episodes = 300, seed = 3, snapshotEvery = 0 } = e.data;
  const data = genEpisodes(episodes, 300, 100 + seed);
  const m = makeModel(seed);
  const tr = new Trainer(m, data, { mode, actionDropout: 0.2, sigmaMax: 0.5, sigmaPow: 3, seed: 50 + seed });
  if (snapshotEvery) self.postMessage({ type: 'snap', mode, step: 0, model: packMLP(m) });
  const curve = [];
  for (let i = 0; i < steps; i += 50) {
    const lr = i < steps * 0.6 ? 1e-3 : i < steps * 0.85 ? 4e-4 : 1.5e-4;
    const l = tr.train(50, 128, lr); curve.push([i + 50, l]);
    self.postMessage({ type: 'progress', mode, step: i + 50, steps, loss: l });
    if (snapshotEvery && (i + 50) % snapshotEvery === 0) self.postMessage({ type: 'snap', mode, step: i + 50, model: packMLP(m) });
  }
  self.postMessage({ type: 'done', mode, model: packMLP(m, { mode, steps, curve }) });
};
