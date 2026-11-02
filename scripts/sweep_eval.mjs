import { genEpisodes, ARENA, norm, N_ACT } from '../src/lib/toyworld.js';
import { rollout, K } from '../src/lib/worldmodel.js';
import { MLP, unpackMLP, mulberry32 } from '../src/lib/nn.js';
import fs from 'fs';
const files = process.argv.slice(2);
const test = genEpisodes(40, 600, 4242);
function feat(X, o, s0, s1) { const a = norm(s0), b = norm(s1); for (let d = 0; d < 4; d++) { X[o + d] = a[d]; X[o + 4 + d] = (b[d] - a[d]) * 8; } X[o + 8] = a[8] - a[0]; X[o + 9] = a[9] - a[1]; }
const FD=10;
function ap(scores, labels) { const idx = scores.map((s, i) => i).sort((i, j) => scores[j] - scores[i]); let tp = 0, sum = 0, P = labels.filter(Boolean).length; if (!P) return NaN; idx.forEach((i, r) => { if (labels[i]) { tp++; sum += tp / (r + 1); } }); return sum / P; }
function mAP(probe, pairs) { const n = pairs.length, X = new Float32Array(n * FD); pairs.forEach((p, i) => feat(X, i * FD, p[0], p[1])); const out = probe.forward(X, n); const aps = []; for (let k = 0; k < N_ACT; k++) { const v = ap(Array.from({ length: n }, (_, i) => out[i * N_ACT + k]), pairs.map((p) => p[2] === k)); if (!isNaN(v)) aps.push(v); } return aps.reduce((a, b) => a + b, 0) / aps.length; }
const train = genEpisodes(150, 200, 777);
const probe = new MLP([FD, 96, 96, N_ACT], 5), rng = mulberry32(3);
const pairs = []; for (const e of train) for (let t = 0; t < e.A.length; t++) pairs.push([e.S[t], e.S[t + 1], e.A[t][0]]);
for (let it = 0; it < 2500; it++) { const B = 128, X = new Float32Array(B * FD), Y = new Float32Array(B * N_ACT); for (let b = 0; b < B; b++) { const p = pairs[Math.floor(rng() * pairs.length)]; feat(X, b * FD, p[0], p[1]); Y[b * N_ACT + p[2]] = 1; } probe.trainMSE(X, Y, B, it < 1800 ? 2e-3 : 5e-4); }
const real=[]; for (const e of test) for (let t = K - 1; t < e.A.length; t++) real.push([e.S[t], e.S[t + 1], e.A[t][0]]);
const mR=mAP(probe,real);
for (const f of files) {
  const m = unpackMLP(JSON.parse(fs.readFileSync(f)));
  for (const nz of [0, 0.03, 0.06, 0.1]) {
    let oob=0,n=0,spd=0,div=0; const g=[];
    for (const e of test){ const acts=e.A.slice(K-1); const r=rollout(m,e.S.slice(0,K),acts,nz,9); const st=[e.S[K-1],...r]; let d=false;
      for(let h=0;h<acts.length;h++){const s=st[h+1]; const ok=s.every(v=>Number.isFinite(v)&&Math.abs(v)<50); if(!(Math.abs(s[8])<ARENA.w+0.1&&Math.abs(s[9])<ARENA.h+0.1&&Math.abs(s[0])<ARENA.w+0.1&&Math.abs(s[4])<ARENA.w+0.1))oob++; if(Math.abs(s[8])>3)d=true; n++; spd+=ok?Math.hypot(s[10],s[11]):1; g.push(ok?[st[h],st[h+1],acts[h][0]]:[s.map(()=>0),s.map(()=>0),acts[h][0]]);} if(d)div++; }
    const ad=test.map(e=>{const r=rollout(m,e.S.slice(0,K),e.A.slice(K-1,K-1+200).map(([a])=>[a,null]),nz,4);return r.reduce((s,x)=>s+Math.hypot(x[4]-x[8],x[5]-x[9]),0)/r.length}).reduce((a,b)=>a+b,0)/test.length;
    console.log(f.split('/').pop(), 'noise',nz,'oob%',(100*oob/n).toFixed(2),'diverged eps',div,'ballspd',(spd/n).toFixed(4),'ARR',(mAP(probe,g)/mR).toFixed(2),'autopilot car2ball',ad.toFixed(3));
  }
}
