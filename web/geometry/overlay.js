// Compute the planar arrangement once when crease sets change. Projection then
// follows the original rigid panels, including any camera-space reorientation.
import { partitionCross } from '../geometry.js';
import { center, chart, containsPoint, dot, next, onSegment, sub, triangulate, unit } from './polygons.js';

export function prepareOverlay(pattern, extraCreases) {
  const regions = partitionCross([...pattern.creases, ...extraCreases]);
  const sourceFrames = pattern.panels.map((panel) => {
    const u = unit(sub(panel[1], panel[0])),
      v = [-u[1], u[0]];
    return (p) => {
      const delta = sub(p, panel[0]);
      return [dot(delta, u), dot(delta, v)];
    };
  });
  const regionsByOwner = [];
  const strokes = pattern.panels.map(() => []);
  const strokeKeys = pattern.panels.map(() => new Set());
  const pointKey = (p) => p.map((x) => Math.round(x * 1e8)).join(',');
  const liesAlong = (a, b, c, d) => onSegment(a, c, d) && onSegment(b, c, d);

  for (const region of regions) {
    // The centroid of an ear is strictly interior even on a concave region.
    const sample = center(triangulate(region)[0].map((i) => region[i]));
    const owner = pattern.panels.findIndex((panel) => containsPoint(panel, sample));
    if (owner < 0) throw new Error('A crease crosses outside the target net.');
    const chartPoint = sourceFrames[owner],
      local = region.map(chartPoint);
    regionsByOwner.push({ owner, local });
    const boundary = pattern.panels[owner];
    region.forEach((a, i) => {
      const b = next(region, i);
      if (!extraCreases.some((c) => liesAlong(a, b, c.a, c.b))) return;
      const isBoundary = boundary.some((p, j) => liesAlong(a, b, p, next(boundary, j)));
      const key = [pointKey(a), pointKey(b)].sort().join('|');
      if (strokeKeys[owner].has(key)) return;
      strokeKeys[owner].add(key);
      strokes[owner].push({ isBoundary, pair: [local[i], next(local, i)] });
    });
  }

  return {
    // Watertight export needs every subdivided face, retaining its winding.
    mapPose(pose) {
      const frames = pose.map(chart);
      return regionsByOwner.map(({ owner, local }) => local.map(frames[owner].xyz));
    },
    // The viewport clips each set of 3D segments to its original panel.
    linesByPanel(pose, { includeBoundary = false } = {}) {
      return pose.map((face, i) => {
        const project = chart(face).xyz;
        return strokes[i].filter((s) => includeBoundary || !s.isBoundary).map((s) => s.pair.map(project));
      });
    },
  };
}
