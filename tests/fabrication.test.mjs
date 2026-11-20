import assert from 'node:assert/strict';
import test from 'node:test';
import { ALL_PATTERNS, PATTERNS, getPattern, partitionCross } from '../web/geometry.js';
import { createNetPDF, createSolidSTL, createKitSTL } from '../web/fabrication.js';

import { inspectSTL } from './mesh-helpers.mjs';

test('every original and mirror net exports closed, bed-oriented solids and disconnected kit plates', async () => {
  for (const pattern of ALL_PATTERNS) {
    for (const edgeMm of [20, 40, 60]) {
      const solid = await inspectSTL(createSolidSTL(pattern, { edgeMm }));
      assert.equal(solid.components, 1);
      if (pattern.id.startsWith('Q')) assert.ok(Math.abs(solid.bounds[2][1] - 2.4) < 1e-6);
      const kit = await inspectSTL(createKitSTL(pattern, { edgeMm }));
      assert.equal(kit.components, pattern.panels.length);
      assert.ok(Math.abs(kit.bounds[2][1] - 1.2) < 1e-6);
    }
  }
});

test('an original net and its mirror form a single combined printable kit and two grooved models', async () => {
  for (const original of PATTERNS.filter((p) => p.id !== 'C0')) {
    const mirror = getPattern(original.id + 'm');
    const creases = [...original.creases, ...mirror.creases];
    const combo = { id: original.id + '+' + mirror.id, creases };
    const kit = await inspectSTL(createKitSTL(combo, { edgeMm: 60 }));
    assert.equal(kit.components, partitionCross(creases).length);
    assert.ok(kit.components > original.panels.length);
    for (const target of [original, mirror]) {
      const solid = await inspectSTL(createSolidSTL(target, { extraCreases: creases }));
      assert.equal(solid.components, 1);
    }
  }
});

test('an omitted crease fuses its neighbors, with the selected gap along the hinge', async () => {
  const cube = getPattern('C0'),
    edgeMm = 40;
  const plain = await inspectSTL(createKitSTL(cube, { edgeMm, selected: [] }));
  assert.equal(plain.components, 1);
  for (const c of cube.creases) {
    const split = await inspectSTL(createKitSTL(cube, { edgeMm, selected: [c.id] }));
    assert.equal(split.components, 2);
    const vertical = c.a[0] === c.b[0],
      axis = vertical ? 0 : 1;
    for (const side of [-1, 1]) {
      const wallCoordinate = c.a[axis] * edgeMm + side * 1.6;
      assert.ok(split.points.some((p) => p[2] === 0 && Math.abs(p[axis] - wallCoordinate) < 1e-4));
    }
  }
});

test('overlay grooves follow every target and printable kit size is checked', async () => {
  const overlays = PATTERNS.flatMap((p) => p.creases.map((c) => ({ ...c, id: p.id + c.id })));
  for (const target of PATTERNS) {
    const original = await inspectSTL(createSolidSTL(target));
    const marked = await inspectSTL(createSolidSTL(target, { extraCreases: overlays }));
    assert.equal(marked.components, 1);
    assert.ok(marked.count > original.count);
  }
  const composite = {
    id: 'C0+O1',
    creases: overlays.filter((c) => c.id.startsWith('C0') || c.id.startsWith('O1')),
  };
  const kit = await inspectSTL(createKitSTL(composite, { edgeMm: 40 }));
  assert.ok(kit.components > getPattern('O1').panels.length);
  assert.throws(
    () => createKitSTL({ id: 'all', creases: overlays }, { edgeMm: 60 }),
    /too small.*Increase the square edge or select fewer creases/,
  );
});

test('PDF keeps physical scale, selected folds, matched seams and Japanese glyphs', async () => {
  const pattern = getPattern('T0');
  for (const [paper, lang] of [
    ['A4', 'en'],
    ['Letter', 'ja'],
  ]) {
    const blob = createNetPDF(pattern, { edgeMm: 60, selected: ['F2', 'F5'], paper, lang });
    assert.equal(blob.type, 'application/pdf');
    const src = await blob.text();
    assert.ok(src.startsWith('%PDF-1.5'));
    const startXref = Number(/startxref\n(\d+)/.exec(src)[1]);
    assert.ok(src.slice(startXref).startsWith('xref\n'));
    const objects = [...src.matchAll(/(\d{10}) 00000 n /g)];
    for (const [index, entry] of objects.entries()) {
      assert.ok(src.slice(Number(entry[1])).startsWith(index + 1 + ' 0 obj'));
    }
    const media = /\/MediaBox \[0 0 ([\d.]+) ([\d.]+)\]/.exec(src).slice(1).map(Number);
    const mm = media.map((pt) => (pt * 25.4) / 72);
    const expected = paper === 'A4' ? [210, 297] : [215.9, 279.4];
    assert.ok(mm.every((x, i) => Math.abs(x - expected[i]) < 1e-3));
    assert.ok(src.includes('/PrintScaling /None'));
    // One dashed legend sample in addition to the two selected folds.
    assert.equal([...src.matchAll(/\[5\.102 3\.26\] 0 d/g)].length, 3);
    for (const letter of 'ABCDEFG')
      assert.equal([...src.matchAll(new RegExp('\\(' + letter + '\\) Tj', 'g'))].length, 2);
    assert.equal(src.includes('/FontFile3'), lang === 'ja');
    const outline = /\[\] 0 d ([-.\d ]+m [^\n]+?) h S/.exec(src)[1];
    const coords = [...outline.matchAll(/([-\d.]+) ([-\d.]+) [ml]/g)].map((m) => [
      Number(m[1]),
      Number(m[2]),
    ]);
    for (const [axis, expectedSpan] of [
      [0, 180],
      [1, 240],
    ]) {
      const values = coords.map((p) => p[axis]);
      assert.ok(Math.abs(((Math.max(...values) - Math.min(...values)) * 25.4) / 72 - expectedSpan) < 1e-3);
    }
  }
});

test('combined PDF appends same-scale assembly keys with each target’s selected folds and seven lettered joins', async () => {
  const targets = [getPattern('C0'), getPattern('T0')];
  const creases = targets.flatMap((p) => p.creases.map((c) => ({ ...c, id: p.id + c.id })));
  const merged = { id: 'C0 + T0', creases, seams: [] };
  const selections = { C0: ['F1'], T0: ['F1', 'F3'] };
  for (const lang of ['ja', 'en']) {
    const text = await createNetPDF(merged, {
      edgeMm: 60,
      lang,
      paper: 'Letter',
      targets,
      selections,
    }).text();
    assert.match(text, /\/Count 3/);
    const pages = [...text.matchAll(/\/Contents (\d+) 0 R/g)].map((m) => {
      const obj = text.indexOf('\n' + m[1] + ' 0 obj\n');
      return text.slice(text.indexOf('stream\n', obj), text.indexOf('\nendstream', obj));
    });
    assert.equal(pages.length, 3);
    assert.deepEqual(
      pages.map((s) => (s.match(/\[5\.102 3\.26\] 0 d/g) ?? []).length),
      [creases.length + 1, 2, 3],
    );
    assert.deepEqual(
      pages.map((s) => (s.match(/\(A\) Tj/g) ?? []).length),
      [0, 2, 2],
    );
    const outlines = pages.map((s) => s.match(/\[\] 0 d ([-.\d ]+m [^\n]+?) h S/)[1]);
    assert.equal(new Set(outlines).size, 1);
  }
});
