// Minimal SVG charts.
const NS = 'http://www.w3.org/2000/svg';
export function svgEl(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

// Horizontal bar chart. data: [{label, value, color, note}]
export function barChart(el, { title, data, better = 'lower', fmt = (v) => v, max, width = 420, labelW = 150 }) {
  const rowH = 30, top = 40, left = labelW, right = 54;
  const H = top + data.length * rowH + 8;
  const svg = svgEl('svg', { viewBox: `0 0 ${width} ${H}`, style: 'width:100%;height:auto' });
  svgEl('text', { x: 0, y: 16, fill: '#e6ebf2', 'font-size': 13.5, 'font-weight': 700 }, svg).textContent = title;
  svgEl('text', { x: 0, y: 32, fill: '#93a0b4', 'font-size': 11 }, svg).textContent = better === 'lower' ? 'lower is better' : 'higher is better';
  const M = max ?? Math.max(...data.map((d) => d.value)) * 1.05;
  data.forEach((d, i) => {
    const y = top + i * rowH;
    svgEl('text', { x: left - 8, y: y + 17, fill: '#c8d1de', 'font-size': 12, 'text-anchor': 'end' }, svg).textContent = d.label;
    const w = Math.max(2, ((width - left - right) * d.value) / M);
    svgEl('rect', { x: left, y: y + 5, width: w, height: rowH - 12, rx: 4, fill: d.color || '#5b6b85' }, svg);
    svgEl('text', { x: left + w + 6, y: y + 17, fill: '#e6ebf2', 'font-size': 12, 'font-family': 'ui-monospace,monospace' }, svg).textContent = fmt(d.value);
  });
  el.innerHTML = ''; el.appendChild(svg);
  return svg;
}

// Line chart. series: [{name, color, points:[[x,y],...], dash}]
export function lineChart(el, { title, series, xlabel = '', ylabel = '', logx = false, logy = false, width = 560, height = 260, ymin, ymax, xmin, xmax, vlines = [], legend = true }) {
  const L = 48, R = 14, B = 36;
  // legend layout: put it on its own line(s) under the title
  const legItems = legend ? series.map((s) => ({ s, w: s.name.length * 6.3 + 26 })) : [];
  const legRows = []; { let row = [], wsum = 0; for (const it of legItems) { if (wsum + it.w > width - L - R && row.length) { legRows.push(row); row = []; wsum = 0; } row.push(it); wsum += it.w; } if (row.length) legRows.push(row); }
  const T = 30 + legRows.length * 16;
  let all = series.flatMap((s) => s.points).filter((p) => isFinite(p[0]) && isFinite(p[1]));
  if (!all.length) all = [[xmin ?? 0, ymin ?? 0], [xmax ?? 1, ymax ?? 1]];
  const fx = logx ? Math.log10 : (v) => v, fy = logy ? Math.log10 : (v) => v;
  const x0 = fx(xmin ?? Math.min(...all.map((p) => p[0]))), x1 = fx(xmax ?? Math.max(...all.map((p) => p[0])));
  const y0 = fy(ymin ?? Math.min(0, ...all.map((p) => p[1]))), y1 = fy(ymax ?? Math.max(...all.map((p) => p[1])) * 1.05);
  const X = (v) => L + ((fx(v) - x0) / (x1 - x0 || 1)) * (width - L - R);
  const Y = (v) => height - B - ((fy(Math.max(v, logy ? 1e-9 : -Infinity)) - y0) / (y1 - y0 || 1)) * (height - T - B);
  const svg = svgEl('svg', { viewBox: `0 0 ${width} ${height}`, style: 'width:100%;height:auto' });
  if (title) svgEl('text', { x: L, y: 15, fill: '#e6ebf2', 'font-size': 13, 'font-weight': 700 }, svg).textContent = title;
  // axes + grid
  for (let i = 0; i <= 4; i++) {
    const yv = y0 + ((y1 - y0) * i) / 4, yy = height - B - ((height - T - B) * i) / 4;
    svgEl('line', { x1: L, x2: width - R, y1: yy, y2: yy, stroke: '#263041', 'stroke-width': 1 }, svg);
    const lab = logy ? Math.pow(10, yv) : yv;
    svgEl('text', { x: L - 6, y: yy + 4, fill: '#66728a', 'font-size': 10.5, 'text-anchor': 'end' }, svg).textContent = Math.abs(lab) >= 100 ? lab.toFixed(0) : Math.abs(lab) >= 10 ? lab.toFixed(1) : lab.toFixed(2);
  }
  const xt = logx ? [1, 2, 4, 10, 30, 60, 120, 300, 1000].filter((v) => fx(v) >= x0 - 1e-9 && fx(v) <= x1 + 1e-9) : Array.from({ length: 5 }, (_, i) => (xmin ?? Math.min(...all.map((p) => p[0]))) + (((xmax ?? Math.max(...all.map((p) => p[0]))) - (xmin ?? Math.min(...all.map((p) => p[0])))) * i) / 4);
  xt.forEach((v) => svgEl('text', { x: X(v), y: height - B + 15, fill: '#66728a', 'font-size': 10.5, 'text-anchor': 'middle' }, svg).textContent = Math.abs(v) >= 10 ? Math.round(v) : +v.toFixed(2));
  svgEl('text', { x: (L + width - R) / 2, y: height - 4, fill: '#93a0b4', 'font-size': 11, 'text-anchor': 'middle' }, svg).textContent = xlabel;
  svgEl('text', { x: 12, y: (T + height - B) / 2, fill: '#93a0b4', 'font-size': 11, 'text-anchor': 'middle', transform: `rotate(-90 12 ${(T + height - B) / 2})` }, svg).textContent = ylabel;
  vlines.forEach((v) => { svgEl('line', { x1: X(v.x), x2: X(v.x), y1: T, y2: height - B, stroke: v.color || '#93a0b4', 'stroke-dasharray': '4 4' }, svg); if (v.label) svgEl('text', { x: X(v.x) + 4, y: T + 10, fill: v.color || '#93a0b4', 'font-size': 10.5 }, svg).textContent = v.label; });
  series.forEach((s) => {
    if (!s.points.length) return;
    const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
    svgEl('path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.width || 2.2, 'stroke-dasharray': s.dash || '', 'stroke-linejoin': 'round' }, svg);
    if (s.dots) s.points.forEach((p) => svgEl('circle', { cx: X(p[0]), cy: Y(p[1]), r: 3, fill: s.color }, svg));
  });
  legRows.forEach((row, r) => {
    let lx = L; const ly = 32 + r * 16;
    row.forEach(({ s, w }) => {
      svgEl('rect', { x: lx, y: ly - 6, width: 14, height: 4, rx: 1, fill: s.color }, svg);
      svgEl('text', { x: lx + 18, y: ly, fill: '#c8d1de', 'font-size': 11 }, svg).textContent = s.name;
      lx += w;
    });
  });
  el.innerHTML = ''; el.appendChild(svg);
  return { svg, X, Y };
}
