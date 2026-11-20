import { OUTLINE, partitionCross } from './geometry.js';

const svgNS = 'http://www.w3.org/2000/svg';
const element = (name, attrs) => {
  const el = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  return el;
};
const line = (a, b, attrs) => element('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], ...attrs });
const paint = (color, width) => ({
  stroke: color,
  'stroke-width': width,
  'vector-effect': 'non-scaling-stroke',
  fill: 'none',
});
const points = (poly) => poly.map((p) => p.join(',')).join(' ');

export function drawNet(svg, model, selected, extras, edge, lang, editLine, showSeams = false) {
  svg.replaceChildren();
  svg.setAttribute('viewBox', '-0.55 -0.29 4.16 4.83');
  svg.setAttribute('aria-label', lang === 'ja' ? model.id + ' の展開図' : model.id + ' crease pattern');
  const base = element('polygon', { points: points(OUTLINE), fill: '#f3e8d6' });
  svg.append(base);
  partitionCross([
    ...model.creases.filter((c) => selected.has(c.id)),
    ...extras.flatMap((p) => p.creases),
  ]).forEach((poly, i) => {
    svg.append(
      element('polygon', {
        points: points(poly),
        fill: ['#f1e5d1', '#f8efdf', '#f5e9d7', '#eddfc7'][i % 4],
      }),
    );
  });
  const dashed = (lineData) => {
    const { a, b, color, width, dash } = lineData;
    return line(a, b, { ...paint(color, width), 'stroke-dasharray': dash });
  };
  const active = model.creases.filter((c) => selected.has(c.id));
  if (editLine)
    for (const crease of model.creases.filter((c) => !selected.has(c.id)))
      svg.append(dashed({ ...crease, color: '#99958b', width: 1.25, dash: '2.5 3.5' }));
  for (const extra of extras)
    for (const crease of extra.creases)
      svg.append(dashed({ ...crease, color: '#39826f', width: 1.7, dash: '5 3' }));
  for (const crease of active)
    svg.append(dashed({ ...crease, color: '#b94d2b', width: 1.8, dash: '5.5 3.5' }));
  svg.append(
    element('polygon', { points: points(OUTLINE), ...paint('#464039', 1.5), 'stroke-linejoin': 'round' }),
  );
  // The dimension spans the full width of the three-square cross.
  const measure = paint('#9b9182', 0.85);
  svg.append(
    line([0, 2.12], [0, 4.28], { ...paint('#b3aa9c', 0.7), 'stroke-dasharray': '2.5 4' }),
    line([3, 2.12], [3, 4.28], { ...paint('#b3aa9c', 0.7), 'stroke-dasharray': '2.5 4' }),
  );
  svg.append(line([0, 4.22], [1.02, 4.22], measure), line([1.98, 4.22], [3, 4.22], measure));
  for (const x of [0, 3]) svg.append(line([x - 0.05, 4.27], [x + 0.05, 4.17], measure));
  const label = element('text', {
    x: 1.5,
    y: 4.28,
    'text-anchor': 'middle',
    'font-family': 'DM Sans, sans-serif',
    'font-size': 0.18,
    'font-weight': 510,
    fill: '#766e61',
  });
  label.textContent = Number((edge * 3).toFixed(2)) + ' mm';
  svg.append(label);
  if (showSeams)
    for (const { a, b, label } of model.seams) {
      const x = (a[0] + b[0]) / 2 - (b[1] - a[1]) * 0.13,
        y = (a[1] + b[1]) / 2 + (b[0] - a[0]) * 0.13;
      const text = element('text', {
        x,
        y: y + 0.05,
        'font-size': 0.15,
        'font-family': 'sans-serif',
        'text-anchor': 'middle',
        fill: '#30322c',
      });
      text.textContent = label;
      svg.append(text);
    }
  if (editLine)
    for (const c of model.creases) {
      const hit = line(c.a, c.b, {
        ...paint('transparent', 15),
        'pointer-events': 'stroke',
        cursor: 'pointer',
      });
      const title = element('title', {});
      title.textContent = c.id;
      hit.append(title);
      hit.addEventListener('click', () => editLine(c.id));
      hit.onmouseenter = () => highlightCrease(svg, c);
      hit.onmouseleave = () => highlightCrease(svg, null);
      svg.append(hit);
    }
}

// Split collinear overlaps into disjoint spans. Preserve each endpoint so a
// template containing short and long versions of a hinge has just one mark.
export function combinedNet(base, selected, overlays) {
  if (!overlays.length) return base;
  const input = [...base.creases.filter((c) => selected.has(c.id)), ...overlays.flatMap((p) => p.creases)];
  const creases = [],
    seen = new Set();
  const key = (p) => p.map((x) => Math.round(x * 1e7)).join(',');
  for (const { a, b } of input) {
    const dx = b[0] - a[0],
      dy = b[1] - a[1],
      len2 = dx * dx + dy * dy;
    const t = (p) => ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2;
    const stops = new Map([
      [key(a), a],
      [key(b), b],
    ]);
    for (const other of input)
      for (const p of [other.a, other.b]) {
        if (Math.abs((p[0] - a[0]) * dy - (p[1] - a[1]) * dx) < 1e-8 && t(p) > 0 && t(p) < 1)
          stops.set(key(p), p);
      }
    const ordered = [...stops.values()].sort((p, q) => t(p) - t(q));
    for (let i = 1; i < ordered.length; i++) {
      const ends = [ordered[i - 1], ordered[i]],
        hash = ends.map(key).sort().join('|');
      if (!seen.has(hash)) {
        seen.add(hash);
        creases.push({ id: 'E' + (creases.length + 1), a: ends[0], b: ends[1] });
      }
    }
  }
  return { id: [base.id, ...overlays.map((p) => p.id)].join('+'), creases, seams: [] };
}
export function highlightCrease(svg, crease) {
  svg.querySelector('.hover-crease')?.remove();
  if (!crease) return;
  const group = element('g', { class: 'hover-crease', 'pointer-events': 'none' });
  group.append(line(crease.a, crease.b, { ...paint('#a84121', 4.5), 'stroke-linecap': 'round' }));
  const x = (crease.a[0] + crease.b[0]) / 2,
    y = (crease.a[1] + crease.b[1]) / 2;
  group.append(
    element('rect', { x: x - 0.19, y: y - 0.14, width: 0.38, height: 0.28, rx: 0.05, fill: '#663824' }),
  );
  const text = element('text', {
    x,
    y: y + 0.062,
    'font-size': 0.19,
    fill: 'white',
    'text-anchor': 'middle',
    'font-family': 'sans-serif',
  });
  text.textContent = crease.id;
  group.append(text);
  svg.append(group);
}
export function markMeasurement(svg, key, result) {
  svg.querySelector('.measure-highlight')?.remove();
  if (!key) return;
  const group = element('g', { class: 'measure-highlight', 'pointer-events': 'none' });
  const mark = (p) =>
    element('circle', {
      cx: p[0],
      cy: p[1],
      r: 0.072,
      fill: '#8b301a',
      stroke: 'white',
      'stroke-width': 0.018,
    });
  if (key === 'area')
    group.append(
      element('polygon', {
        points: points(result.smallestPanel),
        fill: '#eb764c',
        'fill-opacity': 0.74,
        ...{ stroke: '#963119', 'stroke-width': 0.025 },
      }),
    );
  if (key === 'spacing') {
    group.append(line(...result.closestPair, paint('#863019', 3)));
    result.closestPair.forEach((p) => group.append(mark(p)));
  }
  if (key === 'angle') {
    const [a, b, c] = result.angleCorner;
    const point = (p) => {
      const len = Math.hypot(p[0] - b[0], p[1] - b[1]),
        k = Math.min(1, 0.5 / len);
      return [b[0] + k * (p[0] - b[0]), b[1] + k * (p[1] - b[1])];
    };
    group.append(
      element('polygon', {
        points: points([point(a), b, point(c)]),
        fill: '#e27b4c',
        stroke: '#8b301a',
        'stroke-width': 0.025,
      }),
      mark(b),
    );
  }
  svg.append(group);
}
