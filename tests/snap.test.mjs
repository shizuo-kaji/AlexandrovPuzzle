import test from 'node:test';
import assert from 'node:assert/strict';
import { createSnapCoupon, createSnapKit } from '../web/snap-kit.js';
import { getPattern, PATTERNS } from '../web/geometry.js';
import { combinedNet } from '../web/net.js';

// Validate the serialized STL, including the join of every stud to its plate
// and the two open through-bores in each independent snap strap.
async function checkShells(blob, expectedCount, expectedEuler, expectedHeight) {
  const raw = new DataView(await blob.arrayBuffer()),
    n = raw.getUint32(80, true);
  assert.equal(raw.byteLength, 84 + 50 * n);
  const edges = new Map(),
    ids = new Map(),
    parents = [];
  let zMin = Infinity,
    zMax = -Infinity,
    volume = 0;
  function vertex(xyz) {
    const k = xyz.join(',');
    if (!ids.has(k)) {
      ids.set(k, parents.length);
      parents.push(parents.length);
    }
    return ids.get(k);
  }
  function root(i) {
    while (parents[i] !== i) {
      parents[i] = parents[parents[i]];
      i = parents[i];
    }
    return i;
  }
  for (let i = 0; i < n; i++) {
    const tri = Array.from({ length: 3 }, (_, j) =>
      Array.from({ length: 3 }, (_, k) => raw.getFloat32(96 + i * 50 + j * 12 + k * 4, true)),
    );
    const verts = tri.map(vertex),
      [a, b, c] = tri;
    const u = b.map((x, k) => x - a[k]),
      v = c.map((x, k) => x - a[k]);
    const normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    assert.ok(Math.hypot(...normal) > 1e-8, 'Degenerate printed face');
    volume += a.reduce((sum, x, k) => sum + x * normal[k], 0) / 6;
    for (let j = 0; j < 3; j++) {
      zMin = Math.min(zMin, tri[j][2]);
      zMax = Math.max(zMax, tri[j][2]);
      const v = verts[j],
        w = verts[(j + 1) % 3],
        k = v < w ? `${v},${w}` : `${w},${v}`;
      const data = edges.get(k) || [0, 0];
      data[0]++;
      data[1] += v < w ? 1 : -1;
      edges.set(k, data);
      parents[root(v)] = root(w);
    }
  }
  for (const [k, [count, direction]] of edges)
    assert.ok(count === 2 && direction === 0, 'Unmatched face on edge ' + k);
  assert.equal(new Set(parents.map((_, i) => root(i))).size, expectedCount);
  assert.equal(ids.size - edges.size + n, expectedCount * expectedEuler);
  assert.ok(volume > 0);
  assert.equal(zMin, 0);
  assert.ok(Math.abs(zMax - expectedHeight) < 1e-5);
}

test('snap calibration pair has two studded plates and one two-eye TPU strap', async () => {
  const kit = createSnapCoupon();
  await checkShells(kit.panels, 2, 2, 3.2);
  await checkShells(kit.links, 1, -2, 0.8);
});

test('all nine original closures can be snapped at full size', async () => {
  for (const p of PATTERNS) {
    const kit = createSnapKit(p, { edgeMm: 60 });
    assert.ok(kit.hingeLinks >= p.creases.length);
    assert.equal(kit.seamLinks, 7);
    await checkShells(kit.panels, p.panels.length, 2, 3.2);
    await checkShells(kit.links, kit.hingeLinks + 7, -2, 0.8);
  }
});

test('multishape panels include reusable stations for all three seam matchings', async () => {
  const patterns = ['O0', 'C0', 'P0'].map(getPattern),
    first = patterns[0];
  const net = combinedNet(first, new Set(first.creases.map((c) => c.id)), patterns.slice(1));
  const kit = createSnapKit(net, { edgeMm: 50, seamPatterns: patterns });
  assert.ok(kit.panelCount > first.panels.length);
  assert.equal(kit.seamLinks, 7);
  await checkShells(kit.panels, kit.panelCount, 2, 3.2);
  await checkShells(kit.links, kit.hingeLinks + 7, -2, 0.8);
  assert.throws(() => createSnapKit(net, { edgeMm: 20, seamPatterns: patterns }), RangeError);
});

test('incomplete nets have only their available fold straps', async () => {
  const kit = createSnapKit(getPattern('C0'), { selected: ['F1'], edgeMm: 40 });
  assert.equal(kit.panelCount, 2);
  assert.equal(kit.seamLinks, 0);
  await checkShells(kit.panels, 2, 2, 3.2);
  await checkShells(kit.links, kit.hingeLinks, -2, 0.8);
});
