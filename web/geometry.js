import { chords, folds } from './geometry/patterns.js';
import { partition } from './geometry/partition.js';

export const OUTLINE = [
  [2, 0],
  [2, 1],
  [3, 1],
  [3, 2],
  [2, 2],
  [2, 3],
  [2, 4],
  [1, 4],
  [1, 3],
  [1, 2],
  [0, 2],
  [0, 1],
  [1, 1],
  [1, 0],
];

const EPS = 1e-8;
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const directedArea = (a, b, p) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
const liesOn = (a, b, p) =>
  Math.abs(directedArea(a, b, p)) < EPS && distance(a, p) + distance(p, b) <= distance(a, b) + EPS;
const onSegment = (points, a, b) =>
  points.filter((p) => liesOn(a, b, p)).sort((p, q) => distance(a, p) - distance(a, q));
const decode = (p) => (typeof p === 'number' ? OUTLINE[p - 1] : p);

function buildPattern(id) {
  const lines = chords[id].map((line) => line.map(decode));
  const points = [...new Map([...OUTLINE, ...lines.flat()].map((p) => [p.join(), p])).values()];
  const perimeter = OUTLINE.flatMap((p, i) => onSegment(points, p, OUTLINE[(i + 1) % 14]).slice(0, -1));
  const panels = [perimeter];
  const creases = [];

  for (const [start, end] of lines) {
    // A long diagonal can touch a reentrant corner. The panel on either side
    // of that corner has a separate physical hinge.
    const stops = onSegment(points, start, end);
    for (let j = 1; j < stops.length; j++) {
      const a = stops[j - 1],
        b = stops[j];
      const index = panels.findIndex((loop) => loop.includes(a) && loop.includes(b));
      if (index < 0) throw new Error('Intersecting chords in ' + id);
      const loop = panels[index];
      const lo = Math.min(loop.indexOf(a), loop.indexOf(b)),
        hi = Math.max(loop.indexOf(a), loop.indexOf(b));
      panels.splice(index, 1, loop.slice(lo, hi + 1), [...loop.slice(hi), ...loop.slice(0, lo + 1)]);
      creases.push({ id: 'F' + (creases.length + 1), a, b });
    }
  }

  // Anchor a panel near the center of the cross. The polygon dual is a tree.
  const centerDistance = (loop) => {
    const mean = loop.reduce((c, p) => [c[0] + p[0] / loop.length, c[1] + p[1] / loop.length], [0, 0]);
    return distance(mean, [1.5, 1.5]);
  };
  const root = panels.reduce((i, p, j) => (centerDistance(p) < centerDistance(panels[i]) ? j : i), 0);
  const hinges = creases.map((line, i) => {
    const touching = panels.flatMap((p, j) => (p.includes(line.a) && p.includes(line.b) ? [j] : []));
    if (touching.length !== 2) throw new Error('Unpaired crease in ' + id);
    line.panels = touching;
    line.angle = folds[id].angles[i];
    return line;
  });
  const tree = [],
    visited = new Set([root]),
    queue = [root];
  for (const parent of queue) {
    for (const hinge of hinges.filter((h) => h.panels.includes(parent))) {
      const child = hinge.panels.find((i) => i !== parent);
      if (visited.has(child)) continue;
      visited.add(child);
      queue.push(child);
      const loop = panels[child];
      // Orient the axis so positive rotation carries the child toward +z.
      const forward = loop[(loop.indexOf(hinge.a) + 1) % loop.length] === hinge.b;
      const [a, b] = forward ? [hinge.a, hinge.b] : [hinge.b, hinge.a];
      tree.push({ parent, child, hinge, a, b });
    }
  }
  const labels = folds[id].labels;
  const seams = OUTLINE.map((a, i) => ({ a, b: OUTLINE[(i + 1) % 14], label: labels[i] }));
  return { id, creases, panels, seams, tree, root };
}

// Orthonormal frame represented by its three column vectors and origin.
const identity = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
  [0, 0, 0],
];
const rotate = (f, p) => [0, 1, 2].map((i) => f[0][i] * p[0] + f[1][i] * p[1] + f[2][i] * p[2]);
const move = (f, p) => rotate(f, p).map((v, i) => v + f[3][i]);

function hingeFrame(parent, a, b, angle) {
  const length = distance(a, b),
    x = (b[0] - a[0]) / length,
    y = (b[1] - a[1]) / length;
  const c = Math.cos(angle),
    s = Math.sin(angle),
    k = 1 - c;
  const basis = [
    [c + k * x * x, k * x * y, -s * y],
    [k * x * y, c + k * y * y, s * x],
    [s * y, -s * x, c],
  ];
  const delta = rotate(basis, [a[0], a[1], 0]).map((v, i) => (a[i] ?? 0) - v);
  return [...basis.map((v) => rotate(parent, v)), move(parent, delta)];
}

export function foldPanels(pattern, progress = 1, selectedIds) {
  const t = Math.max(0, Math.min(1, progress));
  const selected = selectedIds == null ? null : new Set(selectedIds);
  const frames = [];
  frames[pattern.root] = identity;
  for (const { parent, child, hinge, a, b } of pattern.tree) {
    const angle = selected && !selected.has(hinge.id) ? 0 : t * hinge.angle;
    frames[child] = hingeFrame(frames[parent], a, b, angle);
  }
  return pattern.panels.map((loop, i) => loop.map(([x, y]) => move(frames[i], [x, y, 0])));
}

export const PATTERNS = Object.keys(chords).map(buildPattern);

function mirrorPattern(source) {
  // Retain vertex identity across all panels, crease axes, and labelled seams.
  const points = new Map(OUTLINE.map((p, i) => [p, OUTLINE[OUTLINE.length - 1 - i]]));
  function reflect(point) {
    if (!points.has(point)) points.set(point, [3 - point[0], point[1]]);
    return points.get(point);
  }
  const creases = source.creases.map((c) => ({ ...c, a: reflect(c.a), b: reflect(c.b) }));
  const matching = new Map(creases.map((c) => [c.id, c]));
  // Reflection reverses orientation. Reverse each ring and directed axis so
  // that the same fold amount keeps the printed side facing out throughout.
  const panels = source.panels.map((loop) => loop.map(reflect).reverse());
  const tree = source.tree.map(({ parent, child, hinge, a, b }) => ({
    parent,
    child,
    hinge: matching.get(hinge.id),
    a: reflect(b),
    b: reflect(a),
  }));
  const seams = source.seams.map(({ a, b, label }) => ({ a: reflect(b), b: reflect(a), label }));
  seams.sort((first, second) => OUTLINE.indexOf(first.a) - OUTLINE.indexOf(second.a));
  return { id: source.id + 'm', root: source.root, creases, panels, tree, seams };
}

// The centered square-grid cube net is already invariant under reflection.
export const ALL_PATTERNS = [...PATTERNS, ...PATTERNS.filter((p) => p.id !== 'C0').map(mirrorPattern)];
export const getPattern = (id) => ALL_PATTERNS.find((pattern) => pattern.id === id);

// Accepts crease objects ({a, b}) or endpoint pairs, in square units.
export const partitionCross = (segments) => partition(OUTLINE, segments);
