import { OUTLINE, partitionCross } from './geometry.js';
import {
  add,
  sub,
  scale,
  unit,
  dot,
  norm,
  next,
  onSegment,
  containsPoint,
  triangulate,
} from './geometry/polygons.js';
import { binarySTL, inset, simplified, strip } from './fabrication.js';
import earcut from './vendor/earcut.js';

const SETBACK = 5.5;
const SOCKET_RADIUS = 2.15;
const SPACING = 6.4; // Space for two 6 mm wide stitched ribbon ends.
const fractions = [0.5, ...Array.from({ length: 16 }, (_, i) => [(19 - i) / 40, (21 + i) / 40]).flat()];
const failure = () =>
  new RangeError('Magnetic fittings need more room. Increase the square edge or select fewer creases.');
const key = (p) => p.map((x) => Math.round(x * 1e6)).join(',');
const distance = (p, a, b) => {
  const segment = sub(b, a);
  const t = Math.max(0, Math.min(1, dot(sub(p, a), segment) / dot(segment, segment)));
  return norm(sub(p, add(a, scale(segment, t))));
};
const separated = (a, b) =>
  a.panel !== b.panel ||
  (norm(sub(a.center, b.center)) >= SPACING &&
    distance(a.center, b.center, b.landing) >= 5.2 &&
    distance(b.center, a.center, a.landing) >= 5.2);

// Boundary partners can differ between targets. Reversing a pair identifies
// t on one unit edge with 1-t on the other, for their outward-facing surfaces.
function perimeterGroups(targets) {
  const graph = OUTLINE.map(() => new Set());
  for (const target of targets) {
    const pairs = new Map();
    target.seams.forEach(({ label }, i) => {
      if (pairs.has(label)) {
        const j = pairs.get(label);
        graph[i].add(j);
        graph[j].add(i);
      } else pairs.set(label, i);
    });
  }
  const signs = new Map(),
    groups = [];
  for (let i = 0; i < graph.length; i++) {
    if (signs.has(i)) continue;
    const edges = [i];
    let symmetric = false;
    signs.set(i, 1);
    for (const j of edges)
      for (const k of graph[j]) {
        if (signs.has(k)) {
          if (signs.get(k) === signs.get(j)) symmetric = true;
        } else {
          signs.set(k, -signs.get(j));
          edges.push(k);
        }
      }
    groups.push({ edges, signs, symmetric });
  }
  return groups;
}

function layout(model, { edgeMm = 40, selected, targets } = {}) {
  const edge = Number(edgeMm);
  if (!(edge >= 20 && edge <= 60)) throw new RangeError('Square edge must be between 20 and 60 mm.');
  const ids = new Set(selected ?? model.creases.map((c) => c.id));
  const cells = partitionCross(model.creases.filter((c) => ids.has(c.id))).map((loop) =>
    loop.map((p) => scale(p, edge)),
  );
  const plates = cells.map((region) => {
    const outer = simplified(region);
    const outline = inset(outer, 2.2);
    // A reversed or collapsed edge indicates that the 4.4 mm hinge gap ate a tip.
    if (outline.some((p, i) => dot(sub(next(outline, i), p), sub(next(outer, i), outer[i])) < 1e-5))
      throw failure();
    return { outline, sockets: [] };
  });
  function socket(a, b, t, panel) {
    const axis = unit(sub(b, a)),
      foot = add(a, scale(sub(b, a), t));
    const center = add(foot, scale([-axis[1], axis[0]], SETBACK));
    const jaw = plates[panel].outline;
    if (
      !containsPoint(jaw, center) ||
      jaw.some((p, i) => distance(center, p, next(jaw, i)) < SOCKET_RADIUS + 0.9)
    )
      return null;
    return { panel, center, landing: add(foot, scale([-axis[1], axis[0]], 2.2)) };
  }
  const jobs = [],
    edges = new Map();
  for (const [panel, loop] of cells.entries())
    for (const [i, a] of loop.entries()) {
      const b = next(loop, i),
        id = [key(a), key(b)].sort().join('|');
      if (!edges.has(id)) edges.set(id, []);
      edges.get(id).push({ a, b, panel });
    }
  let hingeNumber = 0;
  for (const owners of edges.values()) {
    if (owners.length !== 2) continue;
    const [first, second] = owners;
    const label = 'H' + ++hingeNumber;
    jobs.push(
      fractions.map((t) =>
        [socket(first.a, first.b, t, first.panel), socket(second.a, second.b, 1 - t, second.panel)].map(
          (anchor) => anchor && { ...anchor, label },
        ),
      ),
    );
  }
  const perimeter = OUTLINE.map((p) => scale(p, edge));
  const boundaries = [...edges.values()].filter((list) => list.length === 1).map((list) => list[0]);
  const usableTargets = targets ?? (model.seams?.length ? [model] : []);
  const groups = usableTargets.length ? perimeterGroups(usableTargets) : [];
  for (const { edges: group, signs, symmetric } of groups) {
    jobs.push(
      (symmetric ? [0.5] : fractions).map((t) =>
        group.map((i) => {
          const a = perimeter[i],
            b = next(perimeter, i),
            u = signs.get(i) === 1 ? t : 1 - t;
          const foot = add(a, scale(sub(b, a), u));
          const span = boundaries.find(
            (v) =>
              onSegment(foot, v.a, v.b) &&
              // Avoid placing an anchor directly across a subdivision or crease tip.
              Math.min(norm(sub(foot, v.a)), norm(sub(foot, v.b))) > 3,
          );
          const anchor = span && socket(a, b, u, span.panel);
          return anchor && { ...anchor, label: 'E' + (i + 1) };
        }),
      ),
    );
  }
  const choices = jobs.map((options) =>
    options.filter((set) =>
      set.every((s, i) => s && set.slice(0, i).every((other) => separated(s, other))),
    ),
  );
  const chosen = [];
  let attempts = 0;
  // Forward-check against the new sockets and place the most constrained
  // remaining joint first. This avoids trying irrelevant distant anchors.
  function place(remaining) {
    if (!remaining.length) return true;
    if (++attempts > 80000) return false;
    remaining.sort((a, b) => a.length - b.length);
    const [current, ...rest] = remaining;
    for (const set of current) {
      const possible = rest.map((group) =>
        group.filter((candidate) => candidate.every((p) => set.every((q) => separated(p, q)))),
      );
      if (possible.some((group) => !group.length)) continue;
      chosen.push(...set);
      if (place(possible)) return true;
      chosen.length -= set.length;
    }
    return false;
  }
  if (!place(choices)) throw failure();
  for (const socket of chosen) plates[socket.panel].sockets.push(socket.center);
  return { plates, edge, socketCount: chosen.length, anchors: chosen, targets: usableTargets };
}

function makePlate(tris, { outline, sockets }) {
  const holes = sockets.map((c) =>
    Array.from({ length: 32 }, (_, i) => {
      const angle = ((i + 0.31) * Math.PI) / 16;
      return add(c, scale([Math.cos(angle), Math.sin(angle)], SOCKET_RADIUS));
    }),
  );
  const allPoints = [...outline, ...holes.flat()],
    holesAt = holes.map((_, i) => outline.length + 32 * i);
  const base = outline.map((p) => [...p, 0]),
    lid = allPoints.map((p) => [...p, 2]);
  tris.push(...triangulate(outline).map((t) => t.map((i) => base[i]).reverse()));
  strip(tris, base, lid.slice(0, outline.length));
  const surface = earcut(allPoints.flat(), holesAt);
  for (let i = 0; i < surface.length; i += 3) tris.push(surface.slice(i, i + 3).map((j) => lid[j]));
  for (const hole of holes) {
    const floor = hole.map((p) => [...p, 0.9]);
    // The wall of a blind socket faces into the cavity.
    strip(tris, [...floor].reverse(), hole.map((p) => [...p, 2]).reverse());
    const middle = [...hole.reduce((a, p) => add(a, scale(p, 1 / 32)), [0, 0]), 0.9];
    for (let i = 0; i < 32; i++) tris.push([middle, floor[i], next(floor, i)]);
  }
}

function kitFiles({ plates, edge, socketCount }, name) {
  const tris = [];
  for (const plate of plates) makePlate(tris, plate);
  // The pocketed top is the exterior. Preserve PDF handedness as viewed from
  // above, rather than turning these plates over like the tape-hinge version.
  const bed = tris.map((t) => t.map(([x, y, z]) => [x, 4 * edge - y, z]).reverse());
  return {
    panels: binarySTL(bed, 'Alexandrov Puzzle ' + name + ' magnetic ribbon kit'),
    panelCount: plates.length,
    pocketCount: socketCount,
    bridgeCount: socketCount / 2,
    magnetCount: socketCount * 2,
  };
}

export function createMagnetKit(model, options = {}) {
  const plan = layout(model, options);
  return { ...kitFiles(plan, model.id), assemblySVG: assemblyDiagram(plan, model.id) };
}

export const createMagnetKitSTL = (model, options) => createMagnetKit(model, options).panels;

// Use the actual 4.4 mm gap, socket geometry, and 11 mm bridge pitch.
export function createMagnetCoupon() {
  const plates = [0, 1].map((i) => {
    const x = 2.2 + i * 18;
    return {
      outline: [
        [x, 2.2],
        [x + 13.6, 2.2],
        [x + 13.6, 15.8],
        [x, 15.8],
      ],
      sockets: [[i ? 23.5 : 12.5, 9]],
    };
  });
  return kitFiles({ plates, edge: 9 / 2, socketCount: 2 }, 'fit coupon');
}

// Keep the net at its physical size so the placement sheet also serves as an
// assembly jig. E numbers are fixed outline edges. Each target supplies its
// own pairing of these edges into the A–G seams on the paper template.
function assemblyDiagram({ plates, edge, anchors, targets }, name) {
  const width = Math.max(175, edge * 3 + 16),
    footer = 13 + 4 * edge,
    height = footer + 13 + targets.length * 4;
  const offset = (width - 3 * edge) / 2;
  const f = (n) => Number(n.toFixed(3));
  const xml = (s) =>
    String(s).replace(
      /[&<>"']/g,
      (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' })[c],
    );
  const xy = ([x, y]) => [f(x + offset), f(y + 12)];
  const text = (x, y, s, size = 3, center = false) =>
    `<text x="${f(x)}" y="${f(y)}" font-size="${size}"${center ? ' text-anchor="middle"' : ''}>${xml(s)}</text>`;
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}mm" height="${height}mm" viewBox="0 0 ${width} ${height}"><title>Alexandrov Puzzle • Magnet placement</title><rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" fill="#17282a">`,
    text(6, 5, 'Alexandrov Puzzle  /  磁石の配置 · Magnet placement', 4),
    text(6, 9.5, 'ポケットを上に向ける · Keep sockets facing up', 3),
  ];
  for (const { outline } of plates)
    parts.push(
      `<polygon points="${outline.map((p) => xy(p).join(',')).join(' ')}" fill="#eef0ec" stroke="#475454" stroke-width=".32"/>`,
    );
  const paired = new Map();
  for (const { center, label } of anchors) {
    if (!label.startsWith('H')) continue;
    if (paired.has(label)) {
      const [x1, y1] = xy(paired.get(label)),
        [x2, y2] = xy(center);
      parts.push(
        `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#527c89" stroke-width=".5" stroke-dasharray="1 0.6"/>`,
      );
    } else paired.set(label, center);
  }
  for (const { center, label } of anchors) {
    const [x, y] = xy(center),
      external = label.startsWith('E');
    parts.push(
      `<circle cx="${x}" cy="${y}" r="${SOCKET_RADIUS}" fill="${external ? '#dae5cf' : '#dfedf4'}" stroke="#3a4e51" stroke-width=".18"/>`,
      text(x, y + 0.52, label, 1.63, true),
    );
  }
  parts.push(
    text(6, footer + 1, 'H · 同じ番号をブリッジでつなぐ  /  Join matching H numbers with one bridge.'),
  );
  if (targets.length)
    parts.push(
      text(6, footer + 5.3, 'E · 外周の接合は下表を参照  /  Pair perimeter sockets for the chosen shape.'),
    );
  targets.forEach((target, i) => {
    const seams = new Map();
    target.seams.forEach(({ label }, j) => {
      if (!seams.has(label)) seams.set(label, []);
      seams.get(label).push('E' + (j + 1));
    });
    const pairs = [...seams.entries()]
      .sort()
      .map(([l, p]) => l + ' ' + p.join('–'))
      .join('   ');
    parts.push(text(6, footer + 10 + 4 * i, target.id + '    ' + pairs, 2.65));
  });
  const y = 7.4,
    x = width - 25;
  parts.push(
    text(6, footer + 10 + 4 * targets.length, '設計 · Design  ' + name, 2.65),
    `<path d="M${x} ${y - 1}v2m0 -1h20m0 -1v2" stroke="#17282a" fill="none" stroke-width=".3"/>`,
    text(x + 2, y - 2, '20 mm · 100%', 2.45),
    '</g></svg>',
  );
  return parts.join('\n');
}
