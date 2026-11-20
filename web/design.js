import { ALL_PATTERNS, getPattern } from './geometry.js';

const storageKey = 'alexandrov-puzzle.design.v2';
const defaultTarget = getPattern('O1');
const allIds = (p) => p.creases.map((c) => c.id);
const allFolds = (p) => new Set(allIds(p));
function loadSaved() {
  try {
    // Designs saved before the rename use the same format.
    const saved = localStorage.getItem(storageKey) ?? localStorage.getItem('fold-atlas.design.v2');
    return JSON.parse(saved) || {};
  } catch {
    return {};
  }
}
function readEdits(encoded) {
  const result = {};
  for (const item of (encoded || '').split(',')) {
    const [id, folds] = item.split('~');
    if (getPattern(id) && folds !== undefined) result[id] = folds.split('.');
  }
  return result;
}
function start() {
  const saved = loadSaved(),
    q = new URL(location.href).searchParams,
    shared = q.has('pattern');
  const model = getPattern(shared ? q.get('pattern') : saved.pattern) || defaultTarget;
  const initialOverlays = shared ? (q.get('overlay') || '').split('.') : saved.overlay || ['Q0'];
  const empty = shared ? q.get('empty') === '1' : saved.empty === true;
  const patterns = empty
    ? []
    : [model, ...ALL_PATTERNS.filter((p) => p !== model && initialOverlays.includes(p.id))];
  const edits = shared ? readEdits(q.get('edit')) : saved.edits || {};
  if (shared && q.has('folds')) edits[model.id] = q.get('folds').split('.');
  const selections = new Map(
    ALL_PATTERNS.map((p) => [
      p.id,
      Array.isArray(edits[p.id])
        ? new Set(allIds(p).filter((id) => edits[p.id].includes(id)))
        : allFolds(p),
    ]),
  );
  const edge = Number(shared ? q.get('mm') : saved.edge);
  const progress = shared ? (q.has('pose') ? Number(q.get('pose')) : 100) : saved.progress;
  const lang = q.get('lang') || saved.lang || (navigator.language.startsWith('ja') ? 'ja' : 'en');
  return {
    model,
    patterns,
    selections,
    edge: edge >= 20 && edge <= 60 ? edge : 60,
    progress: empty ? 0 : Number.isFinite(progress) ? Math.max(0, Math.min(100, progress)) : 100,
    lang: lang === 'ja' ? 'ja' : 'en',
    format: 'pdf',
    method: 'tape',
    paper: (shared ? q.get('paper') : saved.paper) === 'Letter' ? 'Letter' : 'A4',
    get selected() {
      return this.patterns.length ? this.selections.get(this.model.id) : new Set();
    },
    set selected(value) {
      this.selections.set(this.model.id, value);
    },
    get overlays() {
      return this.patterns.filter((p) => p !== this.model);
    },
  };
}
export const state = start();
export function linesFor(p) {
  return p.creases.filter((c) => state.selections.get(p.id).has(c.id));
}
export function overlayLayers() {
  return state.overlays.map((p) => ({ ...p, creases: linesFor(p) }));
}
export function full() {
  return state.patterns.length > 0 && state.selected.size === state.model.creases.length;
}
export function togglePattern(p) {
  if (state.patterns.includes(p)) {
    state.patterns = state.patterns.filter((q) => q !== p);
    if (p === state.model && state.patterns.length) state.model = state.patterns[0];
  } else {
    if (!state.patterns.length) {
      state.model = p;
      state.progress = 100;
    }
    state.patterns.push(p);
  }
  if (!state.patterns.length) state.progress = 0;
}
function editedFolds() {
  return Object.fromEntries(
    state.patterns
      .filter((p) => linesFor(p).length !== p.creases.length)
      .map((p) => [p.id, [...state.selections.get(p.id)]]),
  );
}
export function designURL() {
  const url = new URL(location.href);
  url.hash = '';
  url.search = '';
  const q = url.searchParams;
  q.set('pattern', state.model.id);
  q.set('lang', state.lang);
  q.set('mm', state.edge);
  if (!state.patterns.length) q.set('empty', '1');
  if (state.overlays.length) q.set('overlay', state.overlays.map((p) => p.id).join('.'));
  const edits = Object.entries(editedFolds())
    .map(([id, ids]) => id + '~' + ids.join('.'))
    .join(',');
  if (edits) q.set('edit', edits);
  if (state.progress !== 100) q.set('pose', state.progress);
  if (state.paper !== 'A4') q.set('paper', state.paper);
  return url.href;
}
export function persist() {
  const data = {
    pattern: state.model.id,
    empty: !state.patterns.length,
    overlay: state.overlays.map((p) => p.id),
    edits: editedFolds(),
    edge: state.edge,
    lang: state.lang,
    paper: state.paper,
    progress: state.progress,
  };
  try {
    localStorage.setItem(storageKey, JSON.stringify(data));
  } catch {
    /* Storage is optional. */
  }
  if (new URL(location.href).searchParams.has('pattern')) history.replaceState(null, '', designURL());
}
