// "Rocket Pong": a tiny top-down 2-car + ball physics world used as a stand-in for Rocket League.
// The real game engine (here: this file) produces training data; a learned world model must imitate it
// from (state, action) sequences alone — exactly the relationship between Rocket League and MIRA.
import { mulberry32 } from './nn.js';

export const ARENA = { w: 1.0, h: 0.6 }; // x in [-w,w], y in [-h,h]
export const CAR_R = 0.07, BALL_R = 0.055;
export const N_ACT = 5; // 0 idle, 1 up, 2 down, 3 left, 4 right   (W, S, A, D in the UI)
export const ACT_NAMES = ['·', '↑', '↓', '←', '→'];
const ACC = 0.011, CAR_DRAG = 0.9, BALL_DRAG = 0.985, REST = 0.85;
const DIRS = [[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]];

// state: Float32Array(12) = [c1x,c1y,c1vx,c1vy, c2x,c2y,c2vx,c2vy, bx,by,bvx,bvy]
export const STATE_DIM = 12;
export function kickoff(rng = Math.random) {
  const j = () => (rng() - 0.5) * 0.1;
  return Float32Array.from([-0.6, j() * 3, 0, 0, 0.6, j() * 3, 0, 0, j(), j(), (rng() - 0.5) * 0.02, (rng() - 0.5) * 0.02]);
}

function wall(s, o, r, rest) {
  if (s[o] < -ARENA.w + r) { s[o] = -ARENA.w + r; s[o + 2] = Math.abs(s[o + 2]) * rest; }
  if (s[o] > ARENA.w - r) { s[o] = ARENA.w - r; s[o + 2] = -Math.abs(s[o + 2]) * rest; }
  if (s[o + 1] < -ARENA.h + r) { s[o + 1] = -ARENA.h + r; s[o + 3] = Math.abs(s[o + 3]) * rest; }
  if (s[o + 1] > ARENA.h - r) { s[o + 1] = ARENA.h - r; s[o + 3] = -Math.abs(s[o + 3]) * rest; }
}

function collide(s, a, ra, b, rb, massRatio) {
  // a pushes b. massRatio = share of impulse taken by b (1 = b is light)
  const dx = s[b] - s[a], dy = s[b + 1] - s[a + 1];
  const d = Math.hypot(dx, dy), md = ra + rb;
  if (d >= md || d < 1e-6) return;
  const nx = dx / d, ny = dy / d;
  const rel = (s[a + 2] - s[b + 2]) * nx + (s[a + 3] - s[b + 3]) * ny;
  const overlap = md - d;
  s[b] += nx * overlap * massRatio; s[b + 1] += ny * overlap * massRatio;
  s[a] -= nx * overlap * (1 - massRatio); s[a + 1] -= ny * overlap * (1 - massRatio);
  if (rel > 0) {
    const J = rel * (1 + REST);
    s[b + 2] += nx * J * massRatio; s[b + 3] += ny * J * massRatio;
    s[a + 2] -= nx * J * (1 - massRatio); s[a + 3] -= ny * J * (1 - massRatio);
  }
}

// One tick of the "game engine".
export function step(state, a1, a2) {
  const s = Float32Array.from(state);
  for (const [o, a] of [[0, a1], [4, a2]]) {
    s[o + 2] = (s[o + 2] + DIRS[a][0] * ACC) * CAR_DRAG;
    s[o + 3] = (s[o + 3] + DIRS[a][1] * ACC) * CAR_DRAG;
    s[o] += s[o + 2]; s[o + 1] += s[o + 3];
  }
  s[10] *= BALL_DRAG; s[11] *= BALL_DRAG;
  s[8] += s[10]; s[9] += s[11];
  collide(s, 0, CAR_R, 8, BALL_R, 0.8);
  collide(s, 4, CAR_R, 8, BALL_R, 0.8);
  collide(s, 0, CAR_R, 4, CAR_R, 0.5);
  wall(s, 0, CAR_R, 0.3); wall(s, 4, CAR_R, 0.3); wall(s, 8, BALL_R, REST);
  return s;
}

// "Nexto" stand-in: a scripted bot that drives to a point behind the ball, toward the opponent's side.
export function botAction(s, player, rng = Math.random, eps = 0.12) {
  if (rng() < eps) return Math.floor(rng() * N_ACT);
  const o = player === 0 ? 0 : 4, side = player === 0 ? 1 : -1; // player 0 attacks +x
  const tx = s[8] - side * 0.11, ty = s[9];
  // look ahead a few ticks for velocity
  const ex = tx - (s[o] + s[o + 2] * 4), ey = ty - (s[o + 1] + s[o + 3] * 4);
  let best = 0, bv = 0.004;
  for (let a = 1; a < N_ACT; a++) { const v = DIRS[a][0] * ex + DIRS[a][1] * ey; if (v > bv) { bv = v; best = a; } }
  return best;
}

export function genEpisodes(n, len, seed = 7, eps = 0.12) {
  const rng = mulberry32(seed); const eps_ = [];
  for (let e = 0; e < n; e++) {
    let s = kickoff(rng); const S = [s], A = [];
    // each bot has a "mood" so behaviour is varied but still bot-like
    let hold1 = 0, hold2 = 0, a1 = 0, a2 = 0;
    for (let t = 0; t < len; t++) {
      if (hold1-- <= 0) { a1 = botAction(s, 0, rng, eps); hold1 = Math.floor(rng() * 3); }
      if (hold2-- <= 0) { a2 = botAction(s, 1, rng, eps); hold2 = Math.floor(rng() * 3); }
      A.push([a1, a2]); s = step(s, a1, a2); S.push(s);
    }
    eps_.push({ S, A });
  }
  return eps_;
}

// ---- normalisation: positions are O(1); velocities are scaled up so all dims are comparable
export const VSCALE = 8;
export function norm(s) { const z = Float32Array.from(s); for (const o of [0, 4, 8]) { z[o + 2] *= VSCALE; z[o + 3] *= VSCALE; } return z; }
export function denorm(z) { const s = Float32Array.from(z); for (const o of [0, 4, 8]) { s[o + 2] /= VSCALE; s[o + 3] /= VSCALE; } return s; }
