#!/usr/bin/env bash
# Downloads/derives the large binary assets that are not committed (needs curl + ffmpeg).
set -euo pipefail
cd "$(dirname "$0")/.."
ORT=1.31.1
mkdir -p public/models/dinov3-s16 public/ort public/media public/frames/{hero,real,seq} /tmp/mira-assets
# 1. DINOv3 ViT-S/16 ONNX (same family as MIRA's frozen DINOv3-L/16)
for f in config.json preprocessor_config.json onnx/model.onnx onnx/model.onnx_data; do
  curl -sL -o "public/models/dinov3-s16/$(basename $f)" "https://huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX/resolve/main/$f"
done
# 2. onnxruntime-web runtime files
for f in ort.all.min.mjs ort-wasm-simd-threaded.mjs ort-wasm-simd-threaded.wasm ort-wasm-simd-threaded.jsep.mjs ort-wasm-simd-threaded.jsep.wasm ort-wasm-simd-threaded.asyncify.mjs ort-wasm-simd-threaded.asyncify.wasm; do
  curl -sL -o "public/ort/$f" "https://cdn.jsdelivr.net/npm/onnxruntime-web@$ORT/dist/$f"
done
# 3. Official MIRA media from the project blog
for f in $(curl -sL https://mira-wm.com/blog-post/ | grep -oE '/media/[A-Za-z0-9._%-]*\.(mp4|jpg|png)' | sort -u); do
  curl -sL -o "public/media/$(basename $f)" "https://mira-wm.com$f"
done
# 4. Frames used by the demos
ffmpeg -v error -y -i public/media/hero-v2.mp4 -vf fps=20 -q:v 3 /tmp/mira-assets/f_%04d.jpg
for i in 0020 0060 0100 0140 0180 0220 0270 0320; do cp /tmp/mira-assets/f_$i.jpg public/frames/hero/; done
for t in 1.0 2.0 4.5 7.0; do ffmpeg -v error -y -ss $t -i public/media/dataset-sample.mp4 -frames:v 1 -update 1 -q:v 2 public/frames/real/r_${t/./_}.jpg; done
ffmpeg -v error -y -ss 2 -t 9 -i public/media/hero-v2.mp4 -vf "fps=20,scale=512:288" -q:v 5 public/frames/seq/s_%03d.jpg
echo "assets ready"
