import assert from 'node:assert/strict';
import test from 'node:test';
import { PATTERNS, ALL_PATTERNS, getPattern } from '../web/geometry.js';
import { createMagnetKit, createMagnetCoupon } from '../web/magnet-kit.js';
import { inspectSTL } from './mesh-helpers.mjs';

function checkSocketFloorPoints(mesh, count) {
  // Each blind socket has one 32-segment rim and a center on the 0.9 mm floor.
  const floor = mesh.points.filter((p) => Math.abs(p[2] - 0.9) < 1e-5);
  assert.equal(floor.length, 33 * count);
  assert.equal(mesh.bounds[2][1], 2);
}

test('blind magnet pockets make watertight fitted panels for each original and mirrored model', async () => {
  for (const target of ALL_PATTERNS) {
    const kit = createMagnetKit(target, { edgeMm: 60 });
    const mesh = await inspectSTL(kit.panels);
    assert.equal(mesh.components, kit.panelCount);
    assert.equal(kit.pocketCount, 2 * (target.creases.length + 7));
    assert.equal(kit.bridgeCount * 4, kit.magnetCount);
    checkSocketFloorPoints(mesh, kit.pocketCount);
  }
});

test('magnetic coupon uses the production pitch, floor, wall and hinge gap', async () => {
  const kit = createMagnetCoupon(),
    mesh = await inspectSTL(kit.panels);
  assert.equal(mesh.components, 2);
  assert.equal(kit.pocketCount, 2);
  assert.equal(kit.bridgeCount, 1);
  assert.equal(kit.magnetCount, 4);
  checkSocketFloorPoints(mesh, 2);
  const xs = mesh.points
    .filter((p) => p[2] === 0)
    .map((p) => p[0])
    .sort((a, b) => a - b);
  assert.ok(Math.abs(xs[4] - xs[3] - 4.4) < 1e-5);
  const centers = mesh.points.filter(([x, y, z]) => Math.abs(y - 9) < 1e-5 && Math.abs(z - 0.9) < 1e-5);
  assert.equal(centers.length, 2);
  assert.equal(Math.abs(centers[0][0] - centers[1][0]), 11);
});

test('an open design has only its selected hinges; inset and cover clearance reject tiny parts', async () => {
  const cube = getPattern('C0');
  const kit = createMagnetKit(cube, { selected: ['F1'], targets: [], edgeMm: 40 });
  assert.equal(kit.panelCount, 2);
  assert.equal(kit.bridgeCount, 1);
  assert.equal((await inspectSTL(kit.panels)).components, 2);
  assert.throws(() => createMagnetKit(getPattern('T0'), { edgeMm: 20 }), /need more room.*Increase/);
  const full = {
    id: 'all',
    creases: PATTERNS.flatMap((p) => p.creases.map((c) => ({ ...c, id: p.id + c.id }))),
  };
  assert.throws(() => createMagnetKit(full, { edgeMm: 60, targets: PATTERNS }), /need more room/);
});

test('a reusable O0/P1m magnetic net fits all two-target closing seams', async () => {
  const targets = [getPattern('O0'), getPattern('P1m')];
  const creases = targets.flatMap((p) => p.creases.map((c) => ({ ...c, id: p.id + c.id })));
  const kit = createMagnetKit({ id: 'O0+P1m', creases }, { edgeMm: 60, targets });
  assert.equal(kit.panelCount, 16);
  assert.equal(kit.pocketCount, 48);
  assert.equal(kit.magnetCount, 96);
  const mesh = await inspectSTL(kit.panels);
  assert.equal(mesh.components, 16);
  checkSocketFloorPoints(mesh, 48);
  assert.equal((kit.assemblySVG.match(/<circle /g) ?? []).length, 48);
  assert.ok(kit.assemblySVG.includes('H1</text>'));
  assert.ok(kit.assemblySVG.includes('E14</text>'));
  for (const target of targets) {
    const pairs = new Map();
    target.seams.forEach(({ label }, i) => {
      if (!pairs.has(label)) pairs.set(label, []);
      pairs.get(label).push('E' + (i + 1));
    });
    for (const [label, edges] of pairs) assert.ok(kit.assemblySVG.includes(label + ' ' + edges.join('–')));
  }
});
