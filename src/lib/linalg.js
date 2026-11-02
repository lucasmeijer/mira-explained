// Small linear-algebra helpers (PCA via symmetric eigendecomposition).

// Cyclic Jacobi eigendecomposition of a symmetric n×n matrix (Float64Array, row-major).
// Returns { values (desc), vectors: Float64Array n×n with eigenvectors as ROWS }.
export function eigSym(Ain, n) {
  const A = Float64Array.from(Ain), V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
    if (off < 1e-18) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-14) continue;
        const app = A[p * n + p], aqq = A[q * n + q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vpk = V[p * n + k], vqk = V[q * n + k];
          V[p * n + k] = c * vpk - s * vqk; V[q * n + k] = s * vpk + c * vqk;
        }
      }
    }
  }
  const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[b * n + b] - A[a * n + a]);
  const values = new Float64Array(n), vectors = new Float64Array(n * n);
  idx.forEach((j, r) => { values[r] = A[j * n + j]; for (let k = 0; k < n; k++) vectors[r * n + k] = V[j * n + k]; });
  return { values, vectors };
}

// PCA over rows of X (N×D Float32Array). Returns mean, eigen basis, explained-variance.
export function pca(X, N, D) {
  const mean = new Float64Array(D);
  for (let i = 0; i < N; i++) for (let d = 0; d < D; d++) mean[d] += X[i * D + d];
  for (let d = 0; d < D; d++) mean[d] /= N;
  const C = new Float64Array(D * D), row = new Float64Array(D);
  for (let i = 0; i < N; i++) {
    for (let d = 0; d < D; d++) row[d] = X[i * D + d] - mean[d];
    for (let a = 0; a < D; a++) { const ra = row[a]; if (ra === 0) continue; const o = a * D; for (let b = a; b < D; b++) C[o + b] += ra * row[b]; }
  }
  for (let a = 0; a < D; a++) for (let b = a; b < D; b++) { C[a * D + b] /= N - 1; C[b * D + a] = C[a * D + b]; }
  const { values, vectors } = eigSym(C, D);
  const total = values.reduce((s, v) => s + Math.max(0, v), 0);
  return { mean, values, vectors, total, D };
}

// Project rows of X onto the top-k principal components -> N×k
export function project(P, X, N, k) {
  const { mean, vectors, D } = P, Y = new Float32Array(N * k);
  for (let i = 0; i < N; i++) for (let c = 0; c < k; c++) {
    let s = 0; const vo = c * D, xo = i * D;
    for (let d = 0; d < D; d++) s += (X[xo + d] - mean[d]) * vectors[vo + d];
    Y[i * k + c] = s;
  }
  return Y;
}
