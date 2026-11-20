import earcut from './vendor/earcut.js';
import { OUTLINE, partitionCross } from './geometry.js';
import {
  add,
  area,
  containsPoint,
  dot,
  next,
  norm,
  onSegment,
  scale,
  sub,
  unit,
} from './geometry/polygons.js';
import { binarySTL, inset, simplified, strip } from './fabrication.js';

// Dimensions in mm, independent of the square size. All links are interchangeable.
const SETBACK = 4.5,
  INSET = 1.4,
  EYE = 2.9,
  HOLE = 1.65,
  PLATE = 1.2;
const same = (a, b) => norm(sub(a, b)) < 1e-5;
const lerp = (a, b, t) => add(a, scale(sub(b, a), t));
const key = (p) => p.map((x) => Math.round(x * 1e6)).join(',');
const edgeKey = (a, b) => [key(a), key(b)].sort().join('|');
const failure = (why) => new RangeError('Snap layout: ' + why);

function circle(p, r) {
  // Different angular origins avoid exactly collinear diagonals spanning
  // separate round holes when adjacent studs lie on the same horizontal row.
  const phase = 0.04 + 0.07 * Math.sin(p[0] * 2.01 + p[1] * 1.07);
  return Array.from({ length: 24 }, (_, i) => {
    const a = (i * Math.PI) / 12 + phase;
    return [p[0] + r * Math.cos(a), p[1] + r * Math.sin(a)];
  });
}

function cap(rings, z, reverse = false) {
  const points = rings.flat(),
    holes = [];
  let n = rings[0].length;
  for (const ring of rings.slice(1)) {
    holes.push(n);
    n += ring.length;
  }
  const indices = earcut(points.flat(), holes, 2),
    faces = [];
  for (let k = 0; k < indices.length; k += 3) {
    const face = indices.slice(k, k + 3).map((i) => [...points[i], z]);
    faces.push(reverse ? face.reverse() : face);
  }
  return faces;
}

function makePlate(outline, centers) {
  const bottom = outline.map((p) => [...p, 0]),
    top = outline.map((p) => [...p, PLATE]);
  const ring = (p, r, z) => circle(p, r).map((q) => [...q, z]);
  const footprints = centers.map((p) => circle(p, 1.7));
  const tris = [...cap([outline], 0, true), ...cap([outline, ...footprints], PLATE)];
  strip(tris, bottom, top);
  for (const p of centers) {
    let previous = ring(p, 1.7, PLATE);
    // A filleted foot, 1 mm of clearance for the strap, self-supporting
    // tapered shoulder, and insertion chamfer on the head.
    for (const [z, radius] of [
      [1.4, 1.5],
      [2.2, 1.5],
      [2.8, 1.95],
      [3.2, 1.55],
    ]) {
      const nextRing = ring(p, radius, z);
      strip(tris, previous, nextRing);
      previous = nextRing;
    }
    const center = [...p, 3.2];
    previous.forEach((a, i) => tris.push([a, next(previous, i), center]));
  }
  return tris;
}

function linkOutline() {
  const loop = [];
  function arc(cx, start) {
    for (let i = 0; i <= 12; i++) {
      const a = start + (Math.PI * i) / 12;
      loop.push([cx + EYE * Math.cos(a), EYE * Math.sin(a)]);
    }
  }
  arc(SETBACK, -Math.PI / 2);
  loop.push([3.35, 2.7], [1.55, 1.55], [0, 1.35], [-1.55, 1.55], [-3.35, 2.7]);
  arc(-SETBACK, Math.PI / 2);
  loop.push([-3.35, -2.7], [-1.55, -1.55], [0, -1.35], [1.55, -1.55], [3.35, -2.7]);
  return loop;
}

function makeLinks(count) {
  const rings = [
    linkOutline(),
    circle([-SETBACK, 0], HOLE).reverse(),
    circle([SETBACK, 0], HOLE).reverse(),
  ];
  const one = [...cap(rings, 0, true), ...cap(rings, 0.8)];
  for (const r of rings)
    strip(
      one,
      r.map((p) => [...p, 0]),
      r.map((p) => [...p, 0.8]),
    );
  const tris = [],
    cols = Math.min(10, Math.ceil(Math.sqrt(count)));
  for (let i = 0; i < count; i++) {
    const dx = SETBACK + EYE + (i % cols) * 17,
      dy = EYE + Math.floor(i / cols) * 8;
    for (const tri of one) tris.push(tri.map(([x, y, z]) => [x + dx, y + dy, z]));
  }
  return binarySTL(tris, 'Alexandrov Puzzle TPU snap links');
}

const distanceToSegment = (p, a, b) => {
  const d = sub(b, a),
    t = Math.max(0, Math.min(1, dot(sub(p, a), d) / dot(d, d)));
  return norm(sub(p, lerp(a, b, t)));
};

function buildPanels(model, edge, selected) {
  const ids = new Set(selected ?? model.creases.map((c) => c.id));
  return partitionCross(model.creases.filter((c) => ids.has(c.id))).map((poly) => {
    const loop = poly.map((p) => scale(p, edge)),
      simple = simplified(loop),
      outline = inset(simple, INSET);
    const valid =
      area(outline) > 4 &&
      outline.every((a, i) => {
        const old = sub(next(simple, i), simple[i]),
          d = sub(next(outline, i), a);
        return dot(old, d) > 1e-4 && containsPoint(simple, a);
      });
    if (!valid) throw failure('panel too narrow for a recessed edge');
    return { loop, outline, studs: [] };
  });
}

function mount(a, b, t, panel) {
  const direction = unit(sub(b, a)),
    mouth = lerp(a, b, t);
  const center = add(mouth, scale([-direction[1], direction[0]], SETBACK));
  if (!containsPoint(panel.outline, center)) return null;
  if (panel.outline.some((p, i) => distanceToSegment(center, p, next(panel.outline, i)) < EYE + 0.04))
    return null;
  return { panel, center, mouth };
}

function feasible(left, right) {
  if (left.panel !== right.panel) return true;
  // A conical stud and the full width of a fitted strap must clear its
  // neighbours along the route to the edge, not just at its head.
  if (same(left.center, right.center) && same(left.mouth, right.mouth)) return true;
  return (
    norm(sub(left.center, right.center)) >= 2 * EYE + 0.2 &&
    distanceToSegment(left.center, right.center, right.mouth) >= EYE + 1.65 &&
    distanceToSegment(right.center, left.center, left.mouth) >= EYE + 1.65
  );
}

// Resolve short hinges first. Search alternative positions along the paired
// edges when another link or a small corner occupies the nominal midpoint.
function assign(tasks) {
  let visited = 0;
  const available = (station) => station.every((m) => m.panel.studs.every((other) => feasible(m, other)));
  const reuseScore = (pair) =>
    pair.filter((m) => m.panel.studs.some((o) => same(o.center, m.center))).length;
  function search(pending) {
    if (!pending.length) return true;
    if (++visited > 1800) return false;
    let index = -1,
      options;
    for (let i = 0; i < pending.length; i++) {
      const valid = pending[i].stations.filter(available);
      if (!valid.length) return false;
      if (!options || valid.length < options.length) {
        index = i;
        options = valid;
      }
    }
    options.sort((a, b) => reuseScore(b) - reuseScore(a));
    const rest = pending.filter((_, i) => i !== index);
    for (const station of options) {
      const added = [];
      for (const m of station)
        if (!m.panel.studs.some((o) => same(o.center, m.center))) {
          m.panel.studs.push(m);
          added.push(m);
        }
      if (search(rest)) {
        pending[index].chosen = station;
        return true;
      }
      for (const m of added) m.panel.studs.pop();
    }
    return false;
  }
  if (tasks.some((t) => !t.stations.length) || !search(tasks))
    throw failure('insufficient room for all straps and stud heads');
}

const positions = [0.5, ...Array.from({ length: 22 }, (_, i) => [(24 - i) / 50, (26 + i) / 50]).flat()];
function planHinges(panels) {
  const spans = new Map();
  for (const panel of panels)
    panel.loop.forEach((a, i) => {
      const b = next(panel.loop, i),
        key = edgeKey(a, b);
      if (!spans.has(key)) spans.set(key, []);
      spans.get(key).push({ a, b, panel });
    });
  const tasks = [];
  for (const pair of spans.values()) {
    if (pair.length !== 2) continue;
    const [first, second] = pair,
      length = norm(sub(first.b, first.a));
    const stations = [];
    for (const t of positions) {
      if (Math.min(t, 1 - t) * length < EYE) continue;
      const a = mount(first.a, first.b, t, first.panel),
        b = mount(second.a, second.b, 1 - t, second.panel);
      if (a && b) stations.push([a, b]);
    }
    tasks.push({ stations, length });
  }
  return tasks;
}

function seamTasks(panels, targets, edge) {
  const tasks = new Map(),
    tables = [];
  const faces = (seam) =>
    panels.filter((panel) =>
      panel.loop.some(
        (p, i) => onSegment(p, seam.a, seam.b) && onSegment(next(panel.loop, i), seam.a, seam.b),
      ),
    );
  for (const target of targets) {
    const groups = new Map(),
      pairs = [];
    for (const { a, b, label } of target.seams || []) {
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push({ a: scale(a, edge), b: scale(b, edge) });
    }
    for (const [label, pair] of groups) {
      if (pair.length !== 2) throw failure('unpaired perimeter edges');
      const id = pair
        .map((s) => edgeKey(s.a, s.b))
        .sort()
        .join(':');
      pairs.push({ label, id });
      if (tasks.has(id)) continue;
      const neighbors = pair.map(faces),
        stations = [];
      for (const t of positions) {
        if (Math.min(t, 1 - t) * edge < EYE) continue;
        const mounts = pair.map(({ a, b }, i) => {
          const along = i === 0 ? t : 1 - t,
            mouth = lerp(a, b, along);
          for (const panel of neighbors[i]) {
            if (!panel.loop.some((p, j) => onSegment(mouth, p, next(panel.loop, j)))) continue;
            const m = mount(a, b, along, panel);
            if (m) return m;
          }
        });
        if (mounts.every(Boolean) && mounts[0].panel !== mounts[1].panel) stations.push(mounts);
      }
      tasks.set(id, { stations });
    }
    if (pairs.length) tables.push({ target: target.id, pairs });
  }
  return {
    tasks: [...tasks.values()],
    byId: tasks,
    tables,
    count: Math.max(0, ...tables.map((t) => t.pairs.length)),
  };
}

/** Rigid panels and removable TPU links, in two independently printable STLs. */
export function createSnapKit(
  model,
  { edgeMm = 40, selected, seamPattern, seamPatterns, lang = 'en' } = {},
) {
  if (!(edgeMm >= 20 && edgeMm <= 60)) throw failure('square edge must be 20–60 mm');
  const panels = buildPanels(model, edgeMm, selected),
    hinges = planHinges(panels);
  const active = selected == null ? null : new Set(selected);
  const complete = !active || model.creases.every((c) => active.has(c.id));
  const targets =
    seamPatterns ??
    (seamPattern === null ? [] : [seamPattern ?? model].filter(() => complete || seamPattern));
  const seams = seamTasks(panels, targets, edgeMm);
  if (!hinges.length && !seams.count) throw failure('the flat sheet has no active joints');
  assign([...hinges, ...seams.tasks]);
  let hingeLinks = hinges.length;
  // Widely spaced twin straps stabilize long edges against sideways twist.
  for (const h of hinges.filter((h) => h.length >= 25)) {
    const first = h.chosen;
    const extra = h.stations
      .filter(
        (s) =>
          s.every((m) => m.panel.studs.every((o) => feasible(m, o))) &&
          norm(sub(s[0].center, first[0].center)) >= 9,
      )
      .sort((a, b) => norm(sub(b[0].center, first[0].center)) - norm(sub(a[0].center, first[0].center)))[0];
    if (extra) {
      for (const m of extra) m.panel.studs.push(m);
      h.extra = extra;
      hingeLinks++;
    }
  }
  // Looking down at the studded face, this orientation matches the paper PDF.
  const tris = panels
    .flatMap((p) =>
      makePlate(
        p.outline,
        p.studs.map((m) => m.center),
      ),
    )
    .map((t) => t.map(([x, y, z]) => [x, 4 * edgeMm - y, z]).reverse());
  const assemblySVG = placementDrawing(model.id, edgeMm, panels, hinges, seams, lang);
  return {
    panels: binarySTL(tris, model.id + ' snap panels'),
    links: makeLinks(hingeLinks + seams.count),
    assemblySVG,
    hingeLinks,
    seamLinks: seams.count,
    panelCount: panels.length,
  };
}

/** Two plates and one production-size strap, using the same stud profile. */
export function createSnapCoupon() {
  const pieces = [0, 12 + 2 * INSET].flatMap((x, i) => {
    const loop = [
      [x, 0],
      [x + 12, 0],
      [x + 12, 14],
      [x, 14],
    ];
    const center = [i ? x + SETBACK - INSET : x + 12 - SETBACK + INSET, 7];
    return makePlate(loop, [center]);
  });
  return {
    panels: binarySTL(pieces, 'Snap fit coupon pair'),
    links: makeLinks(1),
    hingeLinks: 1,
    seamLinks: 0,
    panelCount: 2,
  };
}

// A single stud may serve several target seam matchings. Keep one numbered
// callout in the flat layout and resolve each shape in its own A–G table row.
function placementDrawing(id, edge, panels, hinges, seams, lang) {
  const ja = lang === 'ja';
  const w = Math.max(160, 3 * edge + 24),
    dy = 24,
    tableY = dy + 4 * edge + 13;
  const h = tableY + 11 + 6 * seams.tables.length,
    dx = (w - 3 * edge) / 2;
  const dec = (x) => Number(x.toFixed(3)),
    esc = (s) => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  const text = (x, y, content, size, extra = '') =>
    `<text x="${dec(x)}" y="${dec(y)}" font-size="${size}" ${extra}>${esc(content)}</text>`;
  const line = (a, b, color, width) =>
    `<path d="M${a.map(dec)}L${b.map(dec)}" fill="none" stroke="${color}" stroke-width="${width}"/>`;
  const circle = (p, r, fill, stroke = 'none') =>
    `<circle cx="${dec(p[0])}" cy="${dec(p[1])}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="0.3"/>`;
  const title = ja ? 'スナップリンク配置図' : 'Snap link placement';
  const svg = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}mm" height="${h}mm" viewBox="0 0 ${w} ${h}" font-family="sans-serif" fill="#242b29"><title>${title} · ${esc(id)}</title><rect width="100%" height="100%" fill="white"/>`,
  ];
  svg.push(text(5, 6, title + ' · ' + id, 4, 'font-weight="bold"'));
  svg.push(
    text(
      5,
      12,
      ja
        ? '突起のある面を上に置き、緑色の線ごとにリンクを取り付けます。'
        : 'Studs face up. Fit one strap across each green line.',
      2.9,
    ),
  );
  svg.push(
    text(
      5,
      16.5,
      ja
        ? '原寸（100%）で印刷。四角の一辺 ' + edge + ' mm。'
        : 'Print at 100%. Square edge ' + edge + ' mm.',
      2.7,
    ),
  );
  svg.push(
    line([w - 25, 13], [w - 5, 13], '#242b29', 0.35),
    line([w - 25, 12], [w - 25, 14], '#242b29', 0.35),
    line([w - 5, 12], [w - 5, 14], '#242b29', 0.35),
    text(w - 15, 11, '20 mm', 2.5, 'text-anchor="middle"'),
  );
  svg.push(`<g transform="translate(${dec(dx)} ${dy})">`);
  for (const panel of panels)
    svg.push(
      `<polygon points="${panel.outline.map((p) => p.map(dec).join(',')).join(' ')}" fill="#faf8f2" stroke="#45443c" stroke-width="0.32"/>`,
    );
  for (const hinge of hinges)
    for (const station of [hinge.chosen, hinge.extra].filter(Boolean)) {
      const [a, b] = station.map((m) => m.center);
      svg.push(
        line(a, b, '#3c8570', 1.9),
        circle(a, 1.7, '#fff', '#236953'),
        circle(b, 1.7, '#fff', '#236953'),
      );
    }
  const unique = new Map();
  for (const task of seams.tasks) for (const m of task.chosen) unique.set(key(m.center), m);
  const outline = OUTLINE.map((p) => scale(p, edge));
  const order = (m) => {
    const i = outline.findIndex((p, j) => onSegment(m.mouth, p, next(outline, j)));
    return i + norm(sub(m.mouth, outline[i])) / edge;
  };
  const mounts = [...unique.values()].sort((a, b) => order(a) - order(b));
  const numbers = new Map(),
    labels = [];
  for (const [i, m] of mounts.entries()) {
    const number = i + 1,
      outward = unit(sub(m.mouth, m.center)),
      sideways = [-outward[1], outward[0]];
    const candidates = [2.9, 5.4, 8.2].flatMap((r) =>
      [0, 2.5, -2.5].map((s) => add(add(m.mouth, scale(outward, r)), scale(sideways, s))),
    );
    const callout =
      candidates.find(
        (p) =>
          labels.every((q) => norm(sub(p, q)) >= 5.5) &&
          !panels.some((panel) => containsPoint(panel.outline, p)),
      ) ?? candidates.at(-1);
    numbers.set(key(m.center), number);
    labels.push(callout);
    svg.push(
      line(m.center, callout, '#39718c', 0.35),
      circle(m.center, 1.65, '#cfdee6', '#39718c'),
      circle(callout, 2.25, '#fff', '#39718c'),
      text(
        callout[0],
        callout[1] + 0.84,
        number,
        2.35,
        'text-anchor="middle" fill="#23516b" font-weight="bold"',
      ),
    );
  }
  svg.push('</g>');
  if (seams.tables.length) {
    svg.push(
      text(
        5,
        tableY,
        ja
          ? '外周の留め方　作る形の行にある番号同士をつなぎます。'
          : 'Perimeter closures · Join the numbered studs in the row for your target.',
        2.9,
        'font-weight="bold"',
      ),
    );
    const start = 24,
      step = (w - start - 5) / 7;
    for (let j = 0; j < 7; j++)
      svg.push(
        text(
          start + step * (j + 0.5),
          tableY + 5.2,
          String.fromCharCode(65 + j),
          2.6,
          'text-anchor="middle" font-weight="bold"',
        ),
      );
    for (const [i, t] of seams.tables.entries()) {
      const y = tableY + 11 + i * 6;
      svg.push(text(5, y, t.target, 3, 'font-weight="bold"'));
      for (const { label, id } of t.pairs) {
        const j = label.charCodeAt(0) - 65,
          pair = seams.byId.get(id).chosen.map((m) => numbers.get(key(m.center)));
        svg.push(text(start + step * (j + 0.5), y, pair.join('–'), 2.8, 'text-anchor="middle"'));
      }
    }
  } else
    svg.push(
      text(
        5,
        tableY,
        ja ? '部分折り用：外周リンクはありません。' : 'Partial fold: internal links only.',
        2.9,
      ),
    );
  svg.push('</svg>');
  return new Blob([svg.join('\n')], { type: 'image/svg+xml' });
}
