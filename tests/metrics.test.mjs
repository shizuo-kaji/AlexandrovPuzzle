import { test } from 'node:test';
import assert from 'node:assert/strict';
import { getPattern, PATTERNS, partitionCross } from '../web/geometry.js';
import { analyzeNet } from '../web/metrics.js';

const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} ≠ ${expected}`);
const diagonalCross = [
  [
    [1, 0],
    [2, 1],
  ],
  [
    [2, 0],
    [1, 1],
  ],
  [
    [1, 1],
    [2, 1],
  ],
];

test('an empty selection measures the single concave Latin cross', () => {
  const m = analyzeNet([], 40);
  assert.equal(m.panelCount, 1);
  assert.equal(m.vertexCount, 14);
  near(m.minAngleDeg, 90);
  near(m.minAreaMm2, 9600);
  near(m.minSpacingMm, 40);
  assert.deepEqual(m.warnings, []);
  const cube = analyzeNet(getPattern('C0').creases, 40);
  assert.equal(cube.panelCount, 6);
  assert.equal(cube.vertexCount, 14);
  near(cube.minAreaMm2, 1600);
});

test('crossings are vertices and equivalent overlapping creases do not change the measurements', () => {
  const lines = [...diagonalCross, ...diagonalCross.map(([a, b]) => ({ a: b, b: a }))];
  const m = analyzeNet(lines, 20);
  assert.equal(m.panelCount, 5);
  assert.equal(m.vertexCount, 15);
  near(m.minAngleDeg, 45);
  near(m.minAreaMm2, 100);
  near(m.minSpacingMm, 20 / Math.sqrt(2));
  assert.deepEqual(
    m.closestPair.find((p) => p[0] === 1.5),
    [1.5, 0.5],
  );
  assert.deepEqual(m.angleCorner[1], m.angleVertex);
  assert.deepEqual(m, analyzeNet(diagonalCross, 20));
});

test('the closest vertices may be across a panel with no edge between them', () => {
  // Two thin oblique bands intersect to form a rhombus. Its short diagonal
  // is shorter than any one segment of the planar graph.
  const width = 0.035;
  const bands = [0.4, 0.6].flatMap((slope) =>
    [-1, 1].map((side) => {
      const offset = (side * width * Math.hypot(1, slope)) / 2;
      return [
        [1, 2.55 - slope / 2 + offset],
        [2, 2.55 + slope / 2 + offset],
      ];
    }),
  );
  const mm = 40,
    m = analyzeNet(bands, mm);
  const difference = Math.atan(0.6) - Math.atan(0.4);
  near(m.minSpacingMm, (mm * width) / Math.cos(difference / 2));
  near(m.minAngleDeg, (difference * 180) / Math.PI);
  const panels = partitionCross(bands);
  for (const panel of panels) {
    panel.forEach(([x, y], i) => {
      const [u, v] = panel[(i + 1) % panel.length];
      assert.ok(mm * Math.hypot(x - u, y - v) > m.minSpacingMm);
    });
  }
});

test('all nine full layouts give known extrema and correctly scaled advisory flags', () => {
  const lines = PATTERNS.flatMap((p) => p.creases),
    m = analyzeNet(lines, 40);
  assert.equal(m.panelCount, 85);
  assert.equal(m.vertexCount, 79);
  near(m.minAngleDeg, (Math.atan(1 / 8) * 180) / Math.PI);
  near(m.minAreaMm2, 40 ** 2 / 728);
  near(m.minSpacingMm, (40 * Math.sqrt(13)) / 91);
  assert.deepEqual(m.warnings, ['angle', 'area', 'spacing']);
  const large = analyzeNet(lines, 160);
  near(large.minAngleDeg, m.minAngleDeg);
  near(large.minAreaMm2, 16 * m.minAreaMm2);
  near(large.minSpacingMm, 4 * m.minSpacingMm);
  assert.deepEqual(large.warnings, ['angle']);
});

test('guide values are inclusive and invalid physical scales are rejected', () => {
  const m = analyzeNet(diagonalCross, 10);
  assert.equal(m.minAreaMm2, m.limits.areaMm2);
  assert.deepEqual(m.warnings, []);
  const limitAngle = (m.limits.angleDeg * Math.PI) / 180;
  const angle = analyzeNet(
    [
      [
        [1, 0],
        [2, Math.tan(limitAngle)],
      ],
    ],
    60,
  );
  near(angle.minAngleDeg, m.limits.angleDeg);
  assert.deepEqual(angle.warnings, []);
  for (const edge of [-2, 0, NaN, Infinity]) assert.throws(() => analyzeNet([], edge), RangeError);
});
