// Subdivide a planar straight-line graph. Cut each segment at intersections,
// including endpoints of collinear overlaps, then walk each bounded face.
const EPS = 1e-9;
const cross = (ax, ay, bx, by) => ax * by - ay * bx;
const coordKey = (p) => p.map((v) => Math.round(v / EPS)).join();
const within = (t) => t >= -EPS && t <= 1 + EPS;

function addIntersections(line, other) {
  const [x, y] = line.a,
    [u, v] = other.a;
  const dx = line.b[0] - x,
    dy = line.b[1] - y;
  const du = other.b[0] - u,
    dv = other.b[1] - v;
  const det = cross(dx, dy, du, dv);
  if (Math.abs(det) > EPS) {
    const t = cross(u - x, v - y, du, dv) / det;
    const s = cross(u - x, v - y, dx, dy) / det;
    if (!within(t) || !within(s)) return;
    const p = [x + dx * t, y + dy * t];
    line.points.push(p);
    other.points.push(p);
  } else if (Math.abs(cross(u - x, v - y, dx, dy)) <= EPS) {
    for (const [first, second] of [
      [line, other],
      [other, line],
    ]) {
      const [ax, ay] = first.a,
        vx = first.b[0] - ax,
        vy = first.b[1] - ay;
      for (const p of [second.a, second.b]) {
        const t = ((p[0] - ax) * vx + (p[1] - ay) * vy) / (vx * vx + vy * vy);
        if (within(t)) first.points.push(p);
      }
    }
  }
}

export function partition(outline, segments) {
  const perimeter = outline.map((a, i) => [a, outline[(i + 1) % outline.length]]);
  const lines = [...perimeter, ...segments.map((s) => (Array.isArray(s) ? s : [s.a, s.b]))]
    .filter(([a, b]) => coordKey(a) !== coordKey(b))
    .map(([a, b]) => ({ a, b, points: [a, b] }));
  for (let i = 0; i < lines.length; i++) {
    for (let j = i + 1; j < lines.length; j++) addIntersections(lines[i], lines[j]);
  }

  const nodes = [],
    ids = new Map(),
    neighbors = [];
  function node(p) {
    const key = coordKey(p);
    if (!ids.has(key)) {
      ids.set(key, nodes.length);
      nodes.push(p);
      neighbors.push(new Set());
    }
    return ids.get(key);
  }
  for (const { a, b, points } of lines) {
    const vx = b[0] - a[0],
      vy = b[1] - a[1];
    points.sort((p, q) => (p[0] - q[0]) * vx + (p[1] - q[1]) * vy);
    const chain = [...new Set(points.map(node))];
    for (let i = 1; i < chain.length; i++) {
      neighbors[chain[i - 1]].add(chain[i]);
      neighbors[chain[i]].add(chain[i - 1]);
    }
  }
  const ordered = neighbors.map((adjacent, origin) => {
    const [x, y] = nodes[origin];
    const bearing = (id) => Math.atan2(nodes[id][1] - y, nodes[id][0] - x);
    return [...adjacent].sort((i, j) => bearing(i) - bearing(j));
  });

  const result = [],
    visited = new Set();
  for (let i = 0; i < nodes.length; i++) {
    for (const neighbor of ordered[i]) {
      let start = i,
        end = neighbor,
        area = 0;
      const loop = [];
      while (!visited.has(start + ',' + end)) {
        visited.add(start + ',' + end);
        const p = nodes[start],
          q = nodes[end];
        loop.push(p);
        area += cross(p[0], p[1], q[0], q[1]);
        const edges = ordered[end];
        const next = edges[(edges.indexOf(start) + edges.length - 1) % edges.length];
        start = end;
        end = next;
      }
      if (area > EPS) result.push(loop);
    }
  }
  return result;
}
