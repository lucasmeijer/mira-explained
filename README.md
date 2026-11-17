# MIRA, explained: an interactive series

An interactive, in-browser explainer of **MIRA: Multiplayer Interactive World Models with Representation Autoencoders**
(General Intuition × Kyutai × Epic Games, [arXiv 2607.05352](https://arxiv.org/abs/2607.05352)). It covers a real-time, 4-player
Rocket League world model, taken apart one component at a time. Most demos are **real models running in your browser**:
a real DINOv3 encoder, flow-matching and decoder models you train yourself, and small learned world models you can play inside.

> Unofficial. Not affiliated with General Intuition, Kyutai or Epic Games. Paper numbers are quoted from the report.

## Chapters

| Ch. | Page | What runs in the browser |
|---|---|---|
| 00 | `index.html` | Clickable pipeline map + animated "one tick" walkthrough |
| 01 | `ch1-codec.html` | **Real DINOv3-S/16** (ONNX, WebGPU/WASM) on 4 player views: PCA colours, cross-view patch similarity, the paper's PCA + pooling bottleneck |
| 02 | `ch2-flow.html` | Live training of a flow-matching model, then progressive self-distillation for 1-step sampling |
| 03 | `ch3-forcing.html` | Physics engine vs. teacher-forced vs. diffusion-forced learned world models on the same action stream |
| 04 | `ch4-transformer.html` | Tokens, attention by hand, space/time attention explorer, AdaLN action conditioning |
| 04b | `ch4b-attention.html` | Attention step by step on real MIRA latent tokens; multi-head/GQA, RoPE, space/time reshape, the full block, how all layers fit together |
| 05 | `ch5-multiplayer.html` | Play inside a learned world model; autopilot via action dropout; live attribution probe |
| 06 | `ch6-realtime.html` | Rolling KV-cache window + a simulator of the 20 fps serving loop |
| 07 | `ch7-evidence.html` | A mini ARR (action recoverability) study + failure modes traced to components |
| 08 | `ch8-training.html` | Training recipe stage by stage, a decoder you train yourself, a world model learning live, compute per part |

## Run it

```bash
npm install
./scripts/fetch_assets.sh   # downloads DINOv3 ONNX weights, onnxruntime-web, official MIRA videos; extracts frames (needs curl + ffmpeg)
npm run dev                 # http://localhost:5173
npm run build               # static site in dist/ (loads DINOv3 + onnxruntime from Hugging Face / jsDelivr CDNs)
```

The large and third-party assets (videos, frames, model weights, WASM runtime) are **not committed**. `fetch_assets.sh` recreates them.

## Honesty notes

- "Rocket Pong" (`src/lib/toyworld.js`) is a 12-number stand-in for MIRA's latent. Its world models are small MLPs
  (`src/lib/worldmodel.js`, trained by `scripts/train_toy.mjs`). They predict the next state directly from (optionally noised)
  context, which isolates the *context-noise* idea of diffusion forcing without flow-matching sampling.
  In this toy, diffusion forcing buys stability at some cost in controllability. The paper reports no such trade-off at scale.
- DINOv3-S/16 replaces MIRA's DINOv3-L/16, using the last layer only.
- Compute numbers marked as estimates in Ch. 08 are back-of-envelope calculations, not figures from the paper.

## Attribution

- Paper, blog videos and frames: © the MIRA authors / General Intuition, Kyutai. Rocket League © Psyonix LLC / Epic Games, Inc.
  The *Rocket Science* dataset is CC BY-NC-SA 4.0. Commercial use of Rocket League content requires permission from Epic Games.
- DINOv3: Meta AI (ONNX export by onnx-community on Hugging Face). onnxruntime-web: Microsoft.
