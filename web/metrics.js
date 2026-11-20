import { partitionCross } from './geometry.js';
import { area, area2, dot, next, sub } from './geometry/polygons.js';

// Workshop guides for handling a pointed tip, a 5 × 5 mm panel, and the
// nominal gap between two beveled kit plates. The kit exporter checks the
// actual inset profile separately.
export const NET_LIMITS = Object.freeze({ angleDeg: 15, areaMm2: 25, spacingMm: 3.2 });

export function analyzeNet(creases, edgeMm) {
  if (!Number.isFinite(edgeMm) || edgeMm <= 0)
    throw new RangeError('Square edge must be a positive number in millimeters');
  const panels = partitionCross(creases);
  let smallestPanel,
    minArea = Infinity,
    angleCorner,
    minAngle = Infinity;
  for (const panel of panels) {
    const panelArea = area(panel);
    if (panelArea < minArea) {
      minArea = panelArea;
      smallestPanel = panel;
    }
    for (let i = 0; i < panel.length; i++) {
      const before = panel[(i + panel.length - 1) % panel.length],
        at = panel[i],
        after = next(panel, i);
      const incoming = sub(at, before),
        outgoing = sub(after, at);
      // The cells have positive signed area. A clockwise turn is therefore a
      // reflex corner, whose interior angle exceeds π.
      const turn = Math.atan2(area2(incoming, outgoing), dot(incoming, outgoing));
      if (Math.abs(turn) <= 1e-10) continue;
      const interior = Math.PI - turn;
      if (interior < minAngle) {
        minAngle = interior;
        angleCorner = [before, at, after];
      }
    }
  }

  // The planar graph shares the same point object across all incident cells.
  // Count the numbered perimeter points, crease endpoints, and intersections
  // once each, including vertices on opposite sides of a narrow region.
  const vertices = [...new Set(panels.flat())];
  let closestPair,
    minSpacing = Infinity;
  for (let i = 0; i < vertices.length; i++) {
    const a = vertices[i];
    for (let j = i + 1; j < vertices.length; j++) {
      const b = vertices[j],
        spacing = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (spacing < minSpacing) {
        minSpacing = spacing;
        closestPair = [a, b];
      }
    }
  }

  const minAngleDeg = (minAngle * 180) / Math.PI;
  const minAreaMm2 = minArea * edgeMm * edgeMm;
  const minSpacingMm = minSpacing * edgeMm;
  const warnings = [];
  // An exact guide value may differ by round-off after unit conversion.
  const below = (value, guide) => value < guide - 1e-10 * guide;
  if (below(minAngleDeg, NET_LIMITS.angleDeg)) warnings.push('angle');
  if (below(minAreaMm2, NET_LIMITS.areaMm2)) warnings.push('area');
  if (below(minSpacingMm, NET_LIMITS.spacingMm)) warnings.push('spacing');
  return {
    panelCount: panels.length,
    vertexCount: vertices.length,
    minAngleDeg,
    minAreaMm2,
    minSpacingMm,
    angleVertex: angleCorner[1],
    angleCorner,
    smallestPanel,
    closestPair,
    warnings,
    limits: NET_LIMITS,
  };
}
