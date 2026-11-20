import { foldPanels } from './geometry.js';
import { prepareOverlay } from './geometry/overlay.js';

const TAU = Math.PI * 2;
const EPS = 1e-7;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const unit = (a) => a.map((x) => x / (Math.hypot(...a) || 1));
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

// Each drawing has a different stationary face. These views keep the
// distinctive facets visible when switching between closed models.
const HOME_CAMERAS = {
  C0: [-1.04, 0.57],
  T0: [1.04, 0.48],
  T1: [2.38, 0.33],
  Q0: [-1.16, 1.02],
  Q1: [-0.62, 1.02],
  P0: [1.06, 0.34],
  P1: [0.64, 0.35],
  O0: [3.55, 0.54],
  O1: [-0.98, 0.38],
};

function homeCamera(pattern) {
  return HOME_CAMERAS[pattern?.id] || HOME_CAMERAS.O0;
}

function cameraBasis(azimuth, elevation) {
  const c = Math.cos(azimuth),
    s = Math.sin(azimuth);
  const h = Math.cos(elevation),
    v = Math.sin(elevation);
  return { right: [-s, c, 0], up: [-c * v, -s * v, h], eye: [c * h, s * h, v] };
}

function normalOf(loop) {
  // Newell's formula works for concave faces and collinear boundary vertices.
  const normal = [0, 0, 0];
  loop.forEach((a, i) => {
    const b = loop[(i + 1) % loop.length];
    normal[0] += (a[1] - b[1]) * (a[2] + b[2]);
    normal[1] += (a[2] - b[2]) * (a[0] + b[0]);
    normal[2] += (a[0] - b[0]) * (a[1] + b[1]);
  });
  return unit(normal);
}

function tabletopPose(pattern, progress, active) {
  const pose = foldPanels(pattern, progress, active);
  if (pattern.id !== 'O0') return pose;
  // Rest the octahedron on its large face. Follow that sheet panel throughout
  // the animation so switching play direction never changes its orientation.
  const base = pose[9],
    x = unit(base[1].map((v, i) => v - base[0][i]));
  const z = normalOf(base),
    y = cross(z, x);
  return pose.map((loop) => loop.map((p) => [dot(p, x), dot(p, y), dot(p, z)]));
}

function onSegment(p, a, b) {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  const x = p[0] - a[0],
    y = p[1] - a[1];
  const along = x * dx + y * dy;
  return Math.abs(x * dy - y * dx) < EPS && along >= -EPS && along <= dx * dx + dy * dy + EPS;
}

function panelEdges(pattern, active) {
  const selected = active == null ? null : new Set(active);
  return pattern.panels.map((loop) =>
    loop.map((p, i) => {
      const q = loop[(i + 1) % loop.length];
      const crease = pattern.creases.find(({ a, b }) => onSegment(p, a, b) && onSegment(q, a, b));
      return !crease || !selected || selected.has(crease.id);
    }),
  );
}

function makePart(points, normal, index) {
  return { points, normal, offset: dot(normal, points[0]), index };
}

function clipHalf(points, distances, side) {
  const result = [];
  for (let i = 0; i < points.length; i++) {
    const j = (i + 1) % points.length,
      p = points[i],
      q = points[j];
    const d = distances[i] * side,
      e = distances[j] * side;
    if (d >= -EPS) result.push(p);
    if ((d > EPS && e < -EPS) || (d < -EPS && e > EPS)) {
      const t = d / (d - e);
      result.push(p.map((x, k) => x + t * (q[k] - x)));
    }
  }
  return result;
}

// An open sheet can have overhanging or intersecting panels. Split on panel
// planes so occlusion stays correct even when their center depths cross.
function partition(parts) {
  if (!parts.length) return null;
  const ref = parts[0],
    coplanar = [ref],
    front = [],
    back = [];
  for (const part of parts.slice(1)) {
    const d = part.points.map((p) => dot(p, ref.normal) - ref.offset);
    const hasFront = d.some((v) => v > EPS),
      hasBack = d.some((v) => v < -EPS);
    if (hasFront && hasBack) {
      front.push({ ...part, points: clipHalf(part.points, d, 1) });
      back.push({ ...part, points: clipHalf(part.points, d, -1) });
    } else if (hasFront) front.push(part);
    else if (hasBack) back.push(part);
    else coplanar.push(part);
  }
  return { normal: ref.normal, coplanar, front: partition(front), back: partition(back) };
}

function visibleOrder(node, eye, parts = []) {
  if (node) {
    const atFront = dot(node.normal, eye) >= 0;
    visibleOrder(atFront ? node.back : node.front, eye, parts);
    // For a double-covered pentagon, show the seams of the layer on top.
    const layers = node.coplanar.slice().sort((a, b) => dot(b.normal, eye) - dot(a.normal, eye));
    parts.push(...layers);
    visibleOrder(atFront ? node.front : node.back, eye, parts);
  }
  return parts;
}

function frame(pattern, pose, width, height, azimuth, elevation, zoom, thumb = false) {
  const basis = cameraBasis(azimuth, elevation),
    home = cameraBasis(...homeCamera(pattern));
  const vertices = pose.flat();
  function bounds(camera) {
    const xs = vertices.map((p) => dot(p, camera.right)),
      ys = vertices.map((p) => -dot(p, camera.up));
    return [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  }
  const [x0, x1, y0, y1] = bounds(basis),
    [u0, u1, v0, v1] = bounds(home);
  const horizontal = thumb ? 0.82 : 0.79,
    vertical = thumb ? 0.78 : 0.7;
  const baseScale = Math.min(
    (width * horizontal) / Math.max(u1 - u0, 0.1),
    (height * vertical) / Math.max(v1 - v0, 0.1),
  );
  const limit = Math.min((width * 0.88) / Math.max(x1 - x0, 0.1), (height * 0.83) / Math.max(y1 - y0, 0.1));
  const scale = Math.min(baseScale, limit) * zoom;
  const ox = width / 2 - (scale * (x1 + x0)) / 2;
  const oy = height * (thumb ? 0.5 : 0.49) - (scale * (y1 + y0)) / 2;
  return { ...basis, project: (p) => [ox + scale * dot(p, basis.right), oy - scale * dot(p, basis.up)] };
}

function faceColor(normal, camera, gray = false) {
  const facing = dot(normal, camera.eye) >= 0 ? 1 : -1;
  const nx = facing * dot(normal, camera.right),
    ny = facing * dot(normal, camera.up),
    nz = Math.abs(dot(normal, camera.eye));
  const key = Math.max(0, -0.62 * nx + 0.52 * ny + 0.59 * nz);
  const light = clamp(0.09 + 0.66 * key + (0.16 * (normal[2] * facing + 1)) / 2 + 0.16 * nz, 0, 1);
  const dark = gray ? [116, 121, 126] : [162, 53, 27];
  const mid = gray ? [185, 190, 192] : [238, 99, 47];
  const bright = gray ? [237, 239, 238] : [255, 176, 122];
  const a = light < 0.52 ? dark : mid,
    b = light < 0.52 ? mid : bright;
  const t = light < 0.52 ? light / 0.52 : (light - 0.52) / 0.48;
  return a.map((v, j) => Math.round(v + (b[j] - v) * t));
}

function cssColor(rgb, delta = 0) {
  return `rgb(${rgb.map((x) => clamp(x + delta, 0, 255)).join(',')})`;
}

function pathOf(points) {
  const path = new Path2D();
  path.moveTo(...points[0]);
  for (const p of points.slice(1)) path.lineTo(...p);
  path.closePath();
  return path;
}

function paperGrain(ctx) {
  const tile = document.createElement('canvas');
  tile.width = tile.height = 144;
  const texture = tile.getContext('2d'),
    pixels = texture.createImageData(144, 144);
  let seed = 94217;
  for (let i = 0; i < pixels.data.length; i += 4) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const color = seed & 0x80000000 ? 255 : 0;
    pixels.data.set([color, color, color, (seed >>> 24) % 9], i);
  }
  texture.putImageData(pixels, 0, 0);
  return ctx.createPattern(tile, 'repeat');
}

function signedArea(points) {
  return (
    points.reduce((sum, p, i) => {
      const q = points[(i + 1) % points.length];
      return sum + p[0] * q[1] - q[0] * p[1];
    }, 0) / 2
  );
}

// Integrate a disk-shaped area light. The contact edge remains crisp while
// vertices further from the table cast a wider penumbra.
function drawShadow(ctx, pose, camera, width, height, surface) {
  const fade = clamp(camera.eye[2] * 5, 0, 1);
  if (fade === 0) return;
  const factor = 0.48,
    w = Math.ceil(width * factor),
    h = Math.ceil(height * factor);
  if (surface.width !== w || surface.height !== h) {
    surface.width = w;
    surface.height = h;
  }
  const shadow = surface.getContext('2d');
  shadow.clearRect(0, 0, w, h);
  const zMin = Math.min(...pose.flat().map((p) => p[2]));
  shadow.fillStyle = 'rgba(64,48,35,0.012)';
  const direction = camera.right.map((r, i) => 0.35 * r + 0.2 * camera.eye[i]);
  const samples = 29;
  for (let i = 0; i < samples; i++) {
    const r = Math.sqrt((i + 0.5) / samples) * 0.53,
      angle = i * 2.399963;
    const dx = direction[0] + r * Math.cos(angle),
      dy = direction[1] + r * Math.sin(angle);
    shadow.beginPath();
    for (const loop of pose) {
      const shape = loop.map(([x, y, z]) => {
        const distance = z - zMin + 0.032;
        return camera.project([x + distance * dx, y + distance * dy, zMin - 0.032]);
      });
      if (signedArea(shape) < 0) shape.reverse();
      shadow.moveTo(shape[0][0] * factor, shape[0][1] * factor);
      for (const p of shape.slice(1)) shadow.lineTo(p[0] * factor, p[1] * factor);
      shadow.closePath();
    }
    shadow.fill();
  }
  ctx.save();
  ctx.globalAlpha = 0.82 * fade;
  ctx.filter = 'blur(2.8px)';
  ctx.drawImage(surface, 0, 0, width, height);
  ctx.restore();
}

function drawModel(ctx, camera, tree, pose, edges, texture, overlayLines) {
  const styles = pose.map((points, index) => {
    const mapped = points.map(camera.project),
      color = faceColor(normalOf(points), camera);
    const xs = mapped.map((p) => p[0]),
      ys = mapped.map((p) => p[1]);
    const gradient = ctx.createLinearGradient(
      Math.min(...xs),
      Math.min(...ys),
      Math.max(...xs) + 0.01,
      Math.max(...ys),
    );
    gradient.addColorStop(0, cssColor(color, 8));
    gradient.addColorStop(0.58, cssColor(color));
    gradient.addColorStop(1, cssColor(color, -6));
    const rim = new Path2D();
    mapped.forEach((p, j) => {
      if (edges[index][j]) {
        rim.moveTo(...p);
        rim.lineTo(...mapped[(j + 1) % mapped.length]);
      }
    });
    const extra = new Path2D();
    for (const [a, b] of overlayLines?.[index] || []) {
      extra.moveTo(...camera.project(a));
      extra.lineTo(...camera.project(b));
    }
    return { gradient, rim, color, extra };
  });
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  for (const part of visibleOrder(tree, camera.eye)) {
    const path = pathOf(part.points.map(camera.project)),
      { gradient, rim, color, extra } = styles[part.index];
    // The narrow same-colour stroke suppresses raster cracks at BSP splits.
    ctx.strokeStyle = cssColor(color);
    ctx.lineWidth = 0.65;
    ctx.stroke(path);
    ctx.fillStyle = gradient;
    ctx.fill(path);
    ctx.save();
    ctx.clip(path);
    ctx.fillStyle = texture;
    ctx.fill(path);
    ctx.strokeStyle = 'rgba(49,92,73,.9)';
    ctx.lineWidth = 1.25;
    ctx.stroke(extra);
    ctx.strokeStyle = 'rgba(255,216,174,.38)';
    ctx.lineWidth = 2.8;
    ctx.stroke(rim);
    ctx.strokeStyle = 'rgba(109,46,27,.78)';
    ctx.lineWidth = 1.35;
    ctx.stroke(rim);
    ctx.restore();
  }
}

/** Populate an SVG preview with the actual assembled geometry. */
export function renderThumbnail(svg, pattern, selected = false) {
  const w = 112,
    h = 78,
    pose = tabletopPose(pattern, 1);
  const camera = frame(pattern, pose, w, h, ...homeCamera(pattern), 1, true);
  const tree = partition(pose.map((points, i) => makePart(points, normalOf(points), i)));
  const nodes = visibleOrder(tree, camera.eye).map((part) => {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    const points = part.points.map((p) =>
      camera
        .project(p)
        .map((v) => v.toFixed(2))
        .join(','),
    );
    path.setAttribute('d', `M${points.join('L')}Z`);
    path.setAttribute('fill', cssColor(faceColor(part.normal, camera, !selected)));
    path.setAttribute('stroke', selected ? '#9d5235' : '#747c7e');
    path.setAttribute('stroke-width', '0.63');
    path.setAttribute('stroke-linejoin', 'round');
    return path;
  });
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  svg.replaceChildren(...nodes);
}

/** Orbit controls and assembly viewer for a transparent, CSS-sized canvas. */
export function createViewport(canvas) {
  const ctx = canvas.getContext('2d'),
    texture = paperGrain(ctx);
  const shadow = document.createElement('canvas'),
    pointers = new Map();
  let pattern,
    progress = 1,
    active,
    pose,
    edges,
    tree,
    overlay,
    overlayLines;
  let [azimuth, elevation] = homeCamera(),
    zoom = 1,
    raf = 0,
    disposed = false;
  let width, height, ratio;
  canvas.style.touchAction = 'none';
  canvas.style.cursor = 'grab';
  if (!canvas.hasAttribute('tabindex')) canvas.tabIndex = 0;

  function draw() {
    raf = 0;
    const rect = canvas.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    if (width < 1 || height < 1) return;
    ratio = Math.min(window.devicePixelRatio || 1, 2.5);
    const w = Math.round(width * ratio),
      h = Math.round(height * ratio);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!pose?.length) return;
    const camera = frame(pattern, pose, width, height, azimuth, elevation, zoom);
    drawShadow(ctx, pose, camera, width, height, shadow);
    drawModel(ctx, camera, tree, pose, edges, texture, overlayLines);
  }
  function schedule() {
    if (!raf && !disposed) raf = requestAnimationFrame(draw);
  }
  function updatePose() {
    if (pattern) {
      pose = tabletopPose(pattern, progress, active);
      tree = partition(pose.map((points, i) => makePart(points, normalOf(points), i)));
      overlayLines = overlay?.linesByPanel(pose, { includeBoundary: true });
    }
    schedule();
  }
  function setModel(next, at = 1, selected) {
    pattern = next;
    overlay = undefined;
    progress = clamp(at, 0, 1);
    active = selected == null ? undefined : new Set(selected);
    edges = panelEdges(pattern, active);
    [azimuth, elevation] = homeCamera(pattern);
    zoom = 1;
    updatePose();
  }
  function resetView() {
    [azimuth, elevation] = homeCamera(pattern);
    zoom = 1;
    schedule();
  }
  function zoomBy(factor) {
    zoom = clamp(zoom * factor, 0.5, 2.5);
    schedule();
  }
  function orbit(dx, dy) {
    azimuth = (azimuth - dx + TAU) % TAU;
    elevation = clamp(elevation + dy, -1.5, 1.5);
    schedule();
  }
  function pointerDown(e) {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
  }
  function pointerMove(e) {
    const past = pointers.get(e.pointerId);
    if (!past) return;
    const next = [e.clientX, e.clientY];
    if (pointers.size === 1) orbit((next[0] - past[0]) * 0.009, (next[1] - past[1]) * 0.009);
    else if (pointers.size === 2) {
      const other = [...pointers].find(([id]) => id !== e.pointerId)[1];
      const dist = (p) => Math.hypot(p[0] - other[0], p[1] - other[1]);
      if (dist(past) > 1 && dist(next) > 1) zoomBy(dist(next) / dist(past));
    }
    pointers.set(e.pointerId, next);
  }
  function pointerUp(e) {
    pointers.delete(e.pointerId);
    if (!pointers.size) canvas.style.cursor = 'grab';
  }
  function wheel(e) {
    e.preventDefault();
    const pixels = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? height || 400 : 1);
    zoomBy(Math.exp(clamp(-pixels * 0.0015, -1, 1)));
  }
  function key(e) {
    const step = 0.12;
    switch (e.key) {
      case 'ArrowLeft':
        orbit(-step, 0);
        break;
      case 'ArrowRight':
        orbit(step, 0);
        break;
      case 'ArrowUp':
        orbit(0, step);
        break;
      case 'ArrowDown':
        orbit(0, -step);
        break;
      case '+':
      case '=':
        zoomBy(1.12);
        break;
      case '-':
        zoomBy(1 / 1.12);
        break;
      case 'Home':
        resetView();
        break;
      default:
        return;
    }
    e.preventDefault();
  }
  const controls = new AbortController(),
    { signal } = controls;
  for (const [type, handler] of Object.entries({
    pointerdown: pointerDown,
    pointermove: pointerMove,
    pointerup: pointerUp,
    pointercancel: pointerUp,
    lostpointercapture: pointerUp,
    wheel,
    dblclick: resetView,
    keydown: key,
  })) {
    canvas.addEventListener(type, handler, { signal, passive: false });
  }
  const observer = new ResizeObserver(schedule);
  observer.observe(canvas);
  window.addEventListener('resize', schedule, { signal });
  return {
    setModel,
    resetView,
    zoomBy,
    setOverlays(creases) {
      overlay = pattern && creases?.length ? prepareOverlay(pattern, creases) : undefined;
      overlayLines = overlay?.linesByPanel(pose, { includeBoundary: true });
      schedule();
    },
    setProgress(value) {
      progress = clamp(value, 0, 1);
      updatePose();
    },
    setCreases(selected) {
      active = selected == null ? undefined : new Set(selected);
      if (pattern) edges = panelEdges(pattern, active);
      updatePose();
    },
    destroy() {
      disposed = true;
      cancelAnimationFrame(raf);
      observer.disconnect();
      controls.abort();
    },
  };
}
