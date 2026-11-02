// Precompute codec latents (frozen DINOv3-S/16 -> 2x2 pooling -> PCA to 32 channels) for the Ch. 08 decoder demo.
import * as ort from 'onnxruntime-node';
import { execSync } from 'child_process';
import fs from 'fs';
import { pca, project } from '../src/lib/linalg.js';
import { packMLP } from '../src/lib/nn.js';
const s = await ort.InferenceSession.create('public/models/dinov3-s16/model.onnx');
const H = 288, W = 512, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
const raw = (file, x, y, w, h) => execSync(`ffmpeg -v error -i ${file} -vf crop=${w}:${h}:${x}:${y},scale=512:288 -f rawvideo -pix_fmt rgb24 -`, { maxBuffer: 1e8 });
async function feats(rgb) {
  const a = new Float32Array(3 * H * W);
  for (let i = 0; i < H * W; i++) for (let c = 0; c < 3; c++) a[c * H * W + i] = (rgb[i * 3 + c] / 255 - mean[c]) / std[c];
  const o = await s.run({ pixel_values: new ort.Tensor('float32', a, [1, 3, H, W]) });
  const f = o.last_hidden_state.data.slice(5 * 384), P = new Float32Array(144 * 384);
  for (let r = 0; r < 9; r++) for (let c = 0; c < 16; c++) for (const [dr, dc] of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
    const io = ((2 * r + dr) * 32 + 2 * c + dc) * 384, oo = (r * 16 + c) * 384;
    for (let d = 0; d < 384; d++) P[oo + d] += f[io + d] / 4;
  }
  return P;
}
const hero = ['f_0020', 'f_0060', 'f_0100', 'f_0140', 'f_0180', 'f_0220', 'f_0270', 'f_0320'];
const real = ['r_1_0', 'r_2_0', 'r_4_5', 'r_7_0'];
const HV = [[0, 0, 512, 288], [512, 0, 512, 288], [0, 288, 512, 288], [512, 288, 512, 288]];
const RV = [[6, 6, 623, 349], [637, 6, 623, 349], [6, 364, 623, 349], [637, 364, 623, 349]];
const views = [];
for (const f of hero) for (const [v, box] of HV.entries()) views.push({ src: `/frames/hero/${f}.jpg`, box, set: ['f_0100', 'f_0270'].includes(f) ? 'heldout' : 'train', feat: await feats(raw(`public/frames/hero/${f}.jpg`, ...box)) });
for (const f of real) for (const box of RV) views.push({ src: `/frames/real/${f}.jpg`, box, set: 'desert', feat: await feats(raw(`public/frames/real/${f}.jpg`, ...box)) });
const trainV = views.filter((v) => v.set === 'train');
const X = new Float32Array(trainV.length * 144 * 384); trainV.forEach((v, i) => X.set(v.feat, i * 144 * 384));
const P = pca(X, trainV.length * 144, 384);
const out = views.map((v) => { const z = project(P, v.feat, 144, 32); return { src: v.src, box: v.box, set: v.set, z: Buffer.from(z.buffer).toString('base64') }; });
fs.writeFileSync('public/models/toy/latents.json', JSON.stringify({ note: 'DINOv3-S/16 last layer, 2x2 pooled, PCA(32) fit on train views', views: out }));
console.log('views', out.length, 'bytes', fs.statSync('public/models/toy/latents.json').size);
