import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ALL_PATTERNS,
  OUTLINE,
  PATTERNS,
  foldPanels,
  getPattern,
  partitionCross,
} from '../web/geometry.js';

const tolerance = 1e-9;
const difference = (a, b) => a.map((x, i) => x - b[i]);
const distance = (a, b) => Math.hypot(...difference(a, b));
const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const near = (value, expected) =>
  assert.ok(Math.abs(value - expected) < tolerance, `${value} ≠ ${expected}`);

function areaVector(face) {
  const area = [0, 0, 0];
  for (let j = 1; j + 1 < face.length; j++) {
    const triangle = cross(difference(face[j], face[0]), difference(face[j + 1], face[0]));
    triangle.forEach((v, i) => (area[i] += v / 2));
  }
  return area;
}
const normal = (face) => {
  const area = areaVector(face);
  return area.map((v) => v / Math.hypot(...area));
};
const onSegment = (p, a, b) => Math.abs(distance(a, p) + distance(p, b) - distance(a, b)) < tolerance;

for (const pattern of ALL_PATTERNS) {
  test(pattern.id + ' is a rigid hinge tree, including partial crease selections', () => {
    const alternate = pattern.creases.filter((line, i) => i % 2 === 0).map((line) => line.id);
    for (const selection of [undefined, [], alternate]) {
      for (const t of [0, 0.17, 0.5, 0.82, 1]) {
        const panels = foldPanels(pattern, t, selection);
        for (const [i, face] of panels.entries()) {
          const flat = pattern.panels[i];
          for (let j = 0; j < face.length; j++) {
            for (let k = j + 1; k < face.length; k++) {
              near(distance(face[j], face[k]), distance(flat[j], flat[k]));
            }
          }
        }
        const normals = panels.map(normal);
        for (const hinge of pattern.creases) {
          const [left, right] = hinge.panels;
          for (const vertex of [hinge.a, hinge.b]) {
            const leftVertex = pattern.panels[left].indexOf(vertex);
            const rightVertex = pattern.panels[right].indexOf(vertex);
            near(distance(panels[left][leftVertex], panels[right][rightVertex]), 0);
          }
          const angle = selection && !selection.includes(hinge.id) ? 0 : t * hinge.angle;
          near(dot(normals[left], normals[right]), Math.cos(angle));
        }
      }
    }
  });

  test(pattern.id + ' closes its seven seams into the intended convex shape', () => {
    const panels = foldPanels(pattern, 1);
    const points = pattern.panels.flatMap((loop, i) => loop.map((xy, j) => ({ xy, xyz: panels[i][j] })));
    const labels = new Set(pattern.seams.map((s) => s.label));
    assert.equal(labels.size, 7);
    for (const label of labels) {
      const pair = pattern.seams.filter((s) => s.label === label);
      assert.equal(pair.length, 2);
      for (const seam of pair) {
        const mate = pair.find((s) => s !== seam);
        for (const { xy, xyz } of points.filter((p) => onSegment(p.xy, seam.a, seam.b))) {
          const t = distance(seam.a, xy) / distance(seam.a, seam.b);
          const target = mate.b.map((x, k) => x + t * (mate.a[k] - x));
          const match = points.find((q) => distance(q.xy, target) < tolerance);
          assert.ok(match, 'Corresponding crease crossing on the opposite seam');
          near(distance(match.xyz, xyz), 0);
        }
      }
    }
    const normals = panels.map(normal),
      distinct = [];
    let volume = 0,
      surfaceArea = 0;
    for (const [i, face] of panels.entries()) {
      const n = normals[i],
        area = areaVector(face);
      if (!distinct.some((v) => distance(v, n) < tolerance)) distinct.push(n);
      for (const q of panels.flat()) assert.ok(dot(difference(q, face[0]), n) >= -tolerance);
      volume -= dot(face[0], area) / 3;
      surfaceArea += Math.hypot(...area);
    }
    const type = pattern.id[0];
    const expectedFaces = { C: 6, T: 4, Q: 2, P: 5, O: 8 };
    assert.equal(distinct.length, expectedFaces[type]);
    near(surfaceArea, 6);
    const volumes = { C: 1, T: 1 / Math.sqrt(3), Q: 0, P: 2 / 3, O: Math.sqrt(2 + 4 * Math.SQRT2) / 3 };
    near(volume, volumes[type]);
  });
}

test('eight mirror variants reflect every pose and carry their seam labels to the opposite side', () => {
  assert.equal(PATTERNS.length, 9);
  assert.equal(new Set(ALL_PATTERNS.map((p) => p.id)).size, 17);
  assert.equal(getPattern('C0m'), undefined);
  for (const original of PATTERNS.filter((p) => p.id !== 'C0')) {
    const mirror = getPattern(original.id + 'm');
    for (const [i, seam] of mirror.seams.entries()) {
      assert.deepEqual(seam.a, OUTLINE[i]);
      assert.deepEqual(seam.b, OUTLINE[(i + 1) % OUTLINE.length]);
      const reflected = original.seams.find((s) => s.b[0] === 3 - seam.a[0] && s.b[1] === seam.a[1]);
      assert.equal(seam.label, reflected.label);
    }
    const odd = original.creases.filter((c, i) => i % 2).map((c) => c.id);
    for (const selected of [undefined, odd]) {
      for (const t of [0, 0.3, 0.8, 1]) {
        const source = foldPanels(original, t, selected);
        const result = foldPanels(mirror, t, selected);
        for (const [i, face] of source.entries()) {
          face.forEach(([x, y, z], j) => near(distance([3 - x, y, z], result[i][face.length - 1 - j]), 0));
        }
      }
    }
  }
});

test('the mixed drawing partitions the entire cross into consistently oriented cells', () => {
  const all = PATTERNS.flatMap((p) => p.creases);
  const mixed = [...all, ...all.map(({ a, b }) => ({ a: b, b: a }))];
  const cases = [
    [[], 1],
    ...ALL_PATTERNS.map((p) => [p.creases, p.panels.length]),
    [all, 85],
    [mixed, 85],
    [ALL_PATTERNS.flatMap((p) => p.creases), 168],
  ];
  const key = (point) => point.map((v) => Math.round(v / tolerance)).join(',');
  for (const [creases, count] of cases) {
    const cells = partitionCross(creases);
    assert.equal(cells.length, count);
    const sides = new Map(),
      vertices = new Set();
    let area = 0;
    for (const loop of cells) {
      let twiceArea = 0;
      assert.equal(new Set(loop.map(key)).size, loop.length, 'Simple cell boundary');
      for (const [j, a] of loop.entries()) {
        const b = loop[(j + 1) % loop.length],
          ka = key(a),
          kb = key(b);
        const edgeKey = [ka, kb].sort().join(':');
        if (!sides.has(edgeKey)) sides.set(edgeKey, []);
        sides.get(edgeKey).push([a, b]);
        vertices.add(ka);
        twiceArea += a[0] * b[1] - a[1] * b[0];
      }
      assert.ok(twiceArea > tolerance);
      area += twiceArea / 2;
    }
    near(area, 6);
    // The only unpaired edges must form the outer perimeter. Each internal
    // segment is shared in opposite directions; Euler's formula checks disks.
    let perimeter = 0;
    for (const owners of sides.values()) {
      const [[a, b]] = owners;
      if (owners.length === 1) {
        assert.ok(
          OUTLINE.some((p, i) => {
            const q = OUTLINE[(i + 1) % OUTLINE.length];
            return onSegment(a, p, q) && onSegment(b, p, q);
          }),
        );
        perimeter += distance(a, b);
      } else {
        assert.equal(owners.length, 2);
        near(distance(a, owners[1][1]), 0);
        near(distance(b, owners[1][0]), 0);
      }
    }
    near(perimeter, 14);
    assert.equal(vertices.size - sides.size + cells.length, 1);
  }
});
