import { foldPanels, partitionCross } from './geometry.js';
import { prepareOverlay } from './geometry/overlay.js';
import {
  EPS,
  add,
  sub,
  scale,
  dot,
  norm,
  unit,
  cross,
  turn,
  next,
  center,
  area,
  onSegment,
  normal,
  chart,
  triangulate,
} from './geometry/polygons.js';
export { createNetPDF } from './net-pdf.js';

const key = (v) => v.map((x) => Math.round(x * 1e6)).join(',');
const pairKey = (a, b) => [key(a), key(b)].sort().join('|');

function edgeLength(edgeMm) {
  const n = Number(edgeMm);
  if (!(n >= 20 && n <= 60)) throw new RangeError('Square edge must be between 20 and 60 mm');
  return n;
}

// Split both sides of every common edge at exactly the same vertices. Seam
// endpoints can lie halfway along a neighboring panel's straight boundary.
function splitEdges(ring, points) {
  return ring.flatMap((a, i) => {
    const b = next(ring, i),
      v = sub(b, a);
    return points
      .filter((p) => onSegment(p, a, b) && norm(sub(p, b)) > EPS)
      .sort((p, q) => dot(sub(p, q), v));
  });
}

function strip(tris, base, top) {
  for (let i = 0; i < base.length; i++) {
    tris.push([base[i], next(base, i), next(top, i)], [base[i], next(top, i), top[i]]);
  }
}

function inset(ring, distance) {
  return ring.map((p, i) => {
    const prev = ring[(i + ring.length - 1) % ring.length],
      after = next(ring, i);
    const a = unit(sub(p, prev)),
      b = unit(sub(after, p));
    const den = 1 + dot(a, b);
    if (den < EPS) throw new Error('Degenerate panel corner.');
    return add(p, scale([-a[1] - b[1], a[0] + b[0]], distance / den));
  });
}

// Limit the inset on small tips where the nominal bevel would eliminate an
// edge. Connected panels can also have re-entrant corners in a partial net.
function bevelFit(
  loop,
  distances,
  scales = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125],
  accept = () => true,
) {
  const dirs = loop.map((a, i) => sub(next(loop, i), a));
  for (const k of scales) {
    const rings = distances.map((d) => inset(loop, d * k));
    const valid = rings.every(
      (r) =>
        r.every((p, i) => dot(sub(next(r, i), p), dirs[i]) > EPS) &&
        r.every((a, i) =>
          r.every((b, j) => {
            if (i === j || j === (i + 1) % r.length || i === (j + 1) % r.length) return true;
            const a1 = next(r, i),
              b1 = next(r, j);
            return turn(a, a1, b) * turn(a, a1, b1) >= 0 || turn(b, b1, a) * turn(b, b1, a1) >= 0;
          }),
        ),
    );
    if (valid && accept(rings)) return rings;
  }
  throw new RangeError(
    'A panel is too small to print and tape. Increase the square edge or select fewer creases.',
  );
}

function deduplicate(rings, tolerance = EPS) {
  const points = [];
  return {
    rings: rings.map((ring) =>
      ring.map((p) => {
        const existing = points.find((v) => norm(sub(v, p)) < tolerance);
        if (existing) return existing;
        points.push(p);
        return p;
      }),
    ),
    points,
  };
}

function simplified(ring) {
  return ring.filter((p, i) => {
    const a = ring[(i + ring.length - 1) % ring.length],
      b = next(ring, i);
    return Math.abs(turn(a, p, b)) > EPS;
  });
}

function binarySTL(tris, name) {
  const buffer = new ArrayBuffer(84 + 50 * tris.length),
    view = new DataView(buffer);
  new Uint8Array(buffer, 0, 80).set(new TextEncoder().encode(name + ' | mm').slice(0, 80));
  view.setUint32(80, tris.length, true);
  let offset = 84;
  for (const tri of tris) {
    const n = cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]));
    if (!(norm(n) > EPS)) throw new Error('A printable facet has zero area.');
    for (const vec of [unit(n), ...tri]) {
      for (const x of vec) {
        view.setFloat32(offset, x, true);
        offset += 4;
      }
    }
    offset += 2;
  }
  return new Blob([buffer], { type: 'model/stl' });
}

// The narrowest caliper width catches long slivers even when their area is
// adequate. A 1.2 mm top accommodates three nominal 0.4 mm extrusion lines.
function caliperWidth(ring) {
  return Math.min(
    ...ring.map((a, i) => {
      const [x, y] = unit(sub(next(ring, i), a));
      const offsets = ring.map((p) => p[0] * y - p[1] * x);
      return Math.max(...offsets) - Math.min(...offsets);
    }),
  );
}

export function createKitSTL(model, { edgeMm = 40, selected } = {}) {
  const edge = edgeLength(edgeMm),
    ids = new Set(selected ?? model.creases.map((c) => c.id));
  const tris = [];
  const panels = partitionCross(model.creases.filter((c) => ids.has(c.id)));
  for (const panel of panels) {
    const outline = simplified(panel).map((p) => scale(p, edge));
    const [base, top] = bevelFit(
      outline,
      [1.6, 2.15],
      [1, 0.875, 0.75],
      ([bottom, lid]) => area(bottom) >= 7.5 && area(lid) >= 3 && caliperWidth(lid) >= 1.2,
    );

    const bottom = base.map((p) => [...p, 0]),
      shoulder = base.map((p) => [...p, 0.6]),
      lid = top.map((p) => [...p, 1.2]);
    tris.push(...triangulate(base).map((t) => t.map((i) => bottom[i]).reverse()));
    strip(tris, bottom, shoulder);
    strip(tris, shoulder, lid);
    tris.push(...triangulate(top).map((t) => t.map((i) => lid[i])));
  }
  return binarySTL(tris, 'Alexandrov Puzzle ' + model.id + ' tape-hinge kit');
}

// Matching panels share the recessed edge; each face rises to a flat plateau
// just inside its outline. The triangular walls make the grooves slicer-visible.
function groovedFace(tris, face, seamPoint) {
  const frame = chart(face),
    xy = face.map(frame.xy);
  const [insetRing] = bevelFit(xy, [0.65]);
  const plateau = insetRing.map(frame.xyz);
  strip(tris, face.map(seamPoint), plateau);
  tris.push(...triangulate(insetRing).map((t) => t.map((i) => plateau[i])));
}

function putOnBed(tris, orient) {
  const n = orient,
    axis = Math.abs(n[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const u = unit(cross(axis, n)),
    v = cross(n, u);
  const projected = tris.map((t) => t.map((p) => [dot(p, u), dot(p, v), dot(p, n)]));
  const min = [0, 1, 2].map((i) => Math.min(...projected.flat().map((p) => p[i])));
  return projected.map((t) => t.map((p) => sub(p, min)));
}

function solidFaces(rings) {
  const welded = deduplicate(rings, 0.001);
  const faces = welded.rings.map((r) => splitEdges(r, welded.points));
  const interior = center(welded.points),
    tris = [];
  let bed,
    largest = 0;
  const inward = (p) => add(p, scale(unit(sub(interior, p)), 0.42));
  for (let face of faces) {
    let n = normal(face);
    if (dot(n, sub(center(face), interior)) < 0) {
      face = [...face].reverse();
      n = scale(n, -1);
    }
    const surface = area(face.map(chart(face).xy));
    if (surface > largest) {
      largest = surface;
      bed = scale(n, -1);
    }
    groovedFace(tris, face, inward);
  }
  return putOnBed(tris, bed);
}

// A doubly covered polygon has two intrinsically distinct sides. Thicken each
// side with its own panel pattern and connect only the shape's outer rim.
function doubledFaces(rings) {
  const frame = chart(rings[0]);
  const projected = rings.map((r) => r.map(frame.xy));
  const welded = deduplicate(projected, 0.001);
  const faces = welded.rings.map((r) => splitEdges(r, welded.points)),
    tris = [],
    rim = new Map();
  for (const face of faces) {
    const top = area(face) > 0,
      z = top ? 2.4 : 0;
    const xyz = face.map((p) => [...p, z]);
    groovedFace(tris, xyz, (p) => [p[0], p[1], z + (top ? -0.32 : 0.32)]);
    face.forEach((a, i) => {
      const b = next(face, i),
        id = (top ? 'up' : 'down') + pairKey(a, b);
      if (rim.has(id)) rim.delete(id);
      else rim.set(id, { a, b, top });
    });
  }
  for (const { a, b, top } of rim.values()) {
    if (!top) continue;
    const l = [...a, 0.32],
      r = [...b, 0.32],
      L = [...a, 2.08],
      R = [...b, 2.08];
    tris.push([l, r, R], [l, R, L]);
  }
  return putOnBed(tris, [0, 0, 1]);
}

export function createSolidSTL(model, { edgeMm = 40, extraCreases = [] } = {}) {
  const edge = edgeLength(edgeMm);
  const target = foldPanels(model, 1);
  const folded = extraCreases.length ? prepareOverlay(model, extraCreases).mapPose(target) : target;
  const rings = folded.map((r) => r.map((p) => scale(p, edge)));
  const tris = model.id.startsWith('Q') ? doubledFaces(rings) : solidFaces(rings);
  return binarySTL(tris, 'Alexandrov Puzzle ' + model.id + ' finished solid');
}

export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob),
    link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

export { binarySTL, inset, simplified, strip };
