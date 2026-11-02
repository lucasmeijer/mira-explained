# MIRA, explained: an interactive series

Explainers for *MIRA: Multiplayer Interactive World Models with Representation Autoencoders*
(General Intuition × Kyutai × Epic Games, [arXiv 2607.05352](https://arxiv.org/abs/2607.05352)), with demos that run in the browser.

```bash
npm install
./scripts/fetch_assets.sh   # DINOv3 ONNX weights, onnxruntime-web, official MIRA videos, frames
npx vite                    # http://localhost:5173
```

| Ch. | Page | What runs in the browser |
|---|---|---|
| 00 | `index.html` | Clickable pipeline map + animated "one tick" walkthrough |
| 01 | `ch1-codec.html` | **Real DINOv3-S/16** (ONNX, WebGPU/WASM) on 4 player views: PCA colours, cross-view patch similarity, the paper's PCA + pooling bottleneck ablation |
| 02 | `ch2-flow.html` | **Live training** of a flow-matching model, then progressive self-distillation (shortcut) for 1-step sampling |
| 03 | `ch3-forcing.html` | Physics engine vs. teacher-forced vs. diffusion-forced **learned** world models, same action stream, plus optional in-browser retraining |
| 04 | `ch4-transformer.html` | Exact 36×16×20 token layout, space/time/full attention explorer, AdaLN action conditioning wired like MIRA (random weights) |
| 05 | `ch5-multiplayer.html` | Play inside the learned world model; autopilot via action dropout; live attribution probe; with/without-keys evaluation |
| 06 | `ch6-realtime.html` | Rolling KV-cache window + a discrete-event simulator of the 20 fps producer/queue/display loop using real MIRA frames |
| 07 | `ch7-evidence.html` | A mini **ARR** study (inverse-dynamics probe trained live) + failure modes traced back to their components |

Honesty notes:
- "Rocket Pong" (`src/lib/toyworld.js`) is a 12-number stand-in for MIRA's latent. Its world models are small MLPs
  (`src/lib/worldmodel.js`) trained by `scripts/train_toy.mjs`. They predict the next state by regression from (optionally noised)
  context, so they isolate the *context-noise* idea of diffusion forcing without flow-matching sampling. In this toy, diffusion forcing
  buys stability at a small cost in controllability. The paper reports no such trade-off at scale.
- DINOv3-S/16 replaces MIRA's DINOv3-L/16 and uses only the last layer, not an average of intermediate blocks.
- Paper numbers are quoted from the report's tables and figures. Videos are from the official blog (mira-wm.com).
