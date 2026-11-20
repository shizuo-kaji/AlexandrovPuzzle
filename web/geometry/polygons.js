export const EPS = 1e-7;
export const add = (a, b) => a.map((v, i) => v + b[i]);
export const sub = (a, b) => a.map((v, i) => v - b[i]);
export const scale = (a, k) => a.map((v) => v * k);
export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
export const norm = (a) => Math.hypot(...a);
export const unit = (a) => scale(a, 1 / norm(a));
export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const area2 = (a, b) => a[0] * b[1] - a[1] * b[0];
export const turn = (a, b, c) => area2(sub(b, a), sub(c, a));
export const next = (ring, i) => ring[(i + 1) % ring.length];
export const center = (ring) => scale(ring.reduce(add), 1 / ring.length);
export const area = (ring) => ring.reduce((s, p, i) => s + area2(p, next(ring, i)), 0) / 2;
export function onSegment(p, a, b) {
  const v = sub(b, a),
    w = sub(p, a),
    len2 = dot(v, v);
  if (len2 < EPS * EPS) return false;
  const t = dot(w, v) / len2;
  return t >= -EPS && t <= 1 + EPS && norm(sub(w, scale(v, t))) < EPS;
}

export function normal(ring) {
  // The translated Newell sum also works for concave and subdivided polygons.
  const o = ring[0];
  return unit(ring.reduce((sum, p, i) => add(sum, cross(sub(p, o), sub(next(ring, i), o))), [0, 0, 0]));
}

export function chart(ring) {
  const o = ring[0],
    u = unit(sub(ring[1], o)),
    n = normal(ring),
    v = cross(n, u);
  return {
    xy: (p) => [dot(sub(p, o), u), dot(sub(p, o), v)],
    xyz: ([x, y]) => add(o, add(scale(u, x), scale(v, y))),
  };
}

export function triangulate(ring) {
  const indices = ring.map((_, i) => i);
  if (area(ring) < 0) indices.reverse();
  const tris = [];
  while (indices.length > 3) {
    const j = indices.findIndex((i, k) => {
      const prev = indices[(k + indices.length - 1) % indices.length],
        after = next(indices, k);
      const a = ring[prev],
        b = ring[i],
        c = ring[after];
      if (turn(a, b, c) <= EPS) return false;
      return !indices.some(
        (t) =>
          t !== prev &&
          t !== i &&
          t !== after &&
          turn(a, b, ring[t]) >= -EPS &&
          turn(b, c, ring[t]) >= -EPS &&
          turn(c, a, ring[t]) >= -EPS,
      );
    });
    if (j < 0) throw new Error('Could not triangulate the panel.');
    tris.push([indices[(j + indices.length - 1) % indices.length], indices[j], next(indices, j)]);
    indices.splice(j, 1);
  }
  tris.push(indices);
  return tris;
}

export function containsPoint(loop, [x, y]) {
  let inside = false;
  for (let i = 0; i < loop.length; i++) {
    const [ax, ay] = loop[i],
      [bx, by] = next(loop, i);
    if (ay > y !== by > y && x < ax + ((bx - ax) * (y - ay)) / (by - ay)) inside = !inside;
  }
  return inside;
}
