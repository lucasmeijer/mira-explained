import { ARENA, CAR_R, BALL_R } from './toyworld.js';

export function arenaMapper(W, H, pad = 14) {
  const s = Math.min((W - 2 * pad) / (2 * ARENA.w), (H - 2 * pad) / (2 * ARENA.h));
  return { X: (x) => W / 2 + x * s, Y: (y) => H / 2 - y * s, s };
}

export function drawArena(g, W, H, state, { title, color = '#93a0b4', trail = [], diverged = false, ghost = null, autopilot = [false, false], tint = null } = {}) {
  const M = arenaMapper(W, H);
  g.fillStyle = '#0a0d13'; g.fillRect(0, 0, W, H);
  // pitch
  g.fillStyle = tint || '#0f1a14'; g.fillRect(M.X(-ARENA.w), M.Y(ARENA.h), 2 * ARENA.w * M.s, 2 * ARENA.h * M.s);
  g.strokeStyle = 'rgba(255,255,255,0.18)'; g.lineWidth = 1.5;
  g.strokeRect(M.X(-ARENA.w), M.Y(ARENA.h), 2 * ARENA.w * M.s, 2 * ARENA.h * M.s);
  g.beginPath(); g.moveTo(M.X(0), M.Y(ARENA.h)); g.lineTo(M.X(0), M.Y(-ARENA.h)); g.stroke();
  g.beginPath(); g.arc(M.X(0), M.Y(0), 0.18 * M.s, 0, 7); g.stroke();
  // goal mouths (decorative)
  g.fillStyle = 'rgba(56,189,248,0.35)'; g.fillRect(M.X(-ARENA.w) - 4, M.Y(0.2), 4, 0.4 * M.s);
  g.fillStyle = 'rgba(251,146,60,0.35)'; g.fillRect(M.X(ARENA.w), M.Y(0.2), 4, 0.4 * M.s);
  // trail of the ball
  if (trail.length > 1) {
    g.strokeStyle = 'rgba(255,255,255,0.18)'; g.lineWidth = 2; g.beginPath();
    trail.forEach((s, i) => (i ? g.lineTo(M.X(s[8]), M.Y(s[9])) : g.moveTo(M.X(s[8]), M.Y(s[9]))));
    g.stroke();
  }
  if (ghost) drawObjects(g, M, ghost, 0.22);
  if (state) drawObjects(g, M, state, 1, autopilot);
  if (title) { g.fillStyle = color; g.font = '700 12px Inter, sans-serif'; g.fillText(title, 10, 16); }
  if (diverged) {
    g.fillStyle = 'rgba(248,113,113,0.16)'; g.fillRect(0, 0, W, H);
    g.fillStyle = '#f87171'; g.font = '800 15px Inter, sans-serif'; g.textAlign = 'center';
    g.fillText('DIVERGED: state left the arena', W / 2, H / 2); g.textAlign = 'left';
  }
}

function drawObjects(g, M, s, alpha = 1, autopilot = [false, false]) {
  g.globalAlpha = alpha;
  const cars = [[0, '#38bdf8'], [4, '#fb923c']];
  cars.forEach(([o, col], i) => {
    const x = M.X(s[o]), y = M.Y(s[o + 1]), r = CAR_R * M.s;
    if (!isFinite(x) || !isFinite(y)) return;
    const ang = Math.atan2(-s[o + 3], s[o + 2]);
    g.save(); g.translate(x, y); if (Math.hypot(s[o + 2], s[o + 3]) > 0.004) g.rotate(ang);
    g.fillStyle = col; roundRect(g, -r * 1.25, -r * 0.8, r * 2.5, r * 1.6, r * 0.45); g.fill();
    g.fillStyle = 'rgba(0,0,0,0.35)'; g.fillRect(r * 0.3, -r * 0.55, r * 0.55, r * 1.1);
    g.restore();
    g.fillStyle = '#0a0d13'; g.font = `700 ${Math.max(9, r * 0.9) | 0}px Inter, sans-serif`; g.textAlign = 'center'; g.fillText(`P${i + 1}`, x, y + 3.5); g.textAlign = 'left';
    if (autopilot[i]) { g.strokeStyle = '#f472b6'; g.setLineDash([3, 3]); g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, r * 1.9, 0, 7); g.stroke(); g.setLineDash([]); }
  });
  const bx = M.X(s[8]), by = M.Y(s[9]);
  if (isFinite(bx) && isFinite(by)) {
    g.fillStyle = '#f5f7fb'; g.beginPath(); g.arc(bx, by, BALL_R * M.s, 0, 7); g.fill();
    g.strokeStyle = 'rgba(0,0,0,0.4)'; g.lineWidth = 1; g.stroke();
  }
  g.globalAlpha = 1;
}
function roundRect(g, x, y, w, h, r) { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); }

export function isDiverged(s) {
  return !s.every(Number.isFinite) || Math.abs(s[8]) > ARENA.w + 0.25 || Math.abs(s[9]) > ARENA.h + 0.25 || Math.abs(s[0]) > ARENA.w + 0.25 || Math.abs(s[4]) > ARENA.w + 0.25;
}

// Keyboard → action for a player. keys: {up,down,left,right}
export function keyAction(pressed, keys) {
  // priority to most recently pressed handled by caller; here simple precedence
  if (pressed.has(keys.up)) return 1;
  if (pressed.has(keys.down)) return 2;
  if (pressed.has(keys.left)) return 3;
  if (pressed.has(keys.right)) return 4;
  return 0;
}
