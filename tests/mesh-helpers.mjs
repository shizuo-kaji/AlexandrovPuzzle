import assert from 'node:assert/strict';

export async function inspectSTL(blob) {
  assert.equal(blob.type, 'model/stl');
  const bytes = new DataView(await blob.arrayBuffer());
  const count = bytes.getUint32(80, true);
  assert.equal(bytes.byteLength, 84 + 50 * count);
  const edges = new Map(),
    parents = new Map(),
    points = new Map();
  let volume = 0;
  function root(key) {
    while (parents.get(key) !== key) key = parents.get(key);
    return key;
  }
  for (let i = 0; i < count; i++) {
    const index = 84 + 50 * i;
    const readVector = (offset) => [0, 4, 8].map((k) => bytes.getFloat32(index + offset + k, true));
    const pts = [12, 24, 36].map(readVector);
    const [a, b, c] = pts;
    const u = b.map((x, k) => x - a[k]),
      v = c.map((x, k) => x - a[k]);
    const normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    assert.ok(Math.hypot(...normal) > 1e-7, 'Degenerate triangle');
    assert.ok(Math.abs(Math.hypot(...readVector(0)) - 1) < 1e-5, 'Invalid normal');
    volume += a.reduce((s, x, k) => s + x * normal[k], 0) / 6;
    const keys = pts.map((p) => p.join(','));
    for (let k = 0; k < 3; k++) {
      points.set(keys[k], pts[k]);
      if (!parents.has(keys[k])) parents.set(keys[k], keys[k]);
    }
    for (let k = 0; k < 3; k++) {
      const start = keys[k],
        end = keys[(k + 1) % 3];
      const id = [start, end].sort().join('|');
      if (!edges.has(id)) edges.set(id, []);
      edges.get(id).push(start);
      parents.set(root(end), root(start));
    }
  }
  for (const owners of edges.values()) {
    assert.equal(owners.length, 2, 'Open or nonmanifold edge');
    assert.notEqual(owners[0], owners[1], 'Inconsistent triangle winding');
  }
  const components = new Set([...points.keys()].map(root)).size;
  assert.equal(points.size - edges.size + count, components * 2, 'Non-spherical shell');
  assert.ok(volume > 0);
  const coords = [...points.values()];
  const bounds = [0, 1, 2].map((i) => [
    Math.min(...coords.map((p) => p[i])),
    Math.max(...coords.map((p) => p[i])),
  ]);
  assert.equal(bounds[2][0], 0, 'Base is not on the build plate');
  return { count, components, bounds, points: coords };
}
