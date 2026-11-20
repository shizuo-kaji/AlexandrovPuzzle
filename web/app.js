import { ALL_PATTERNS, getPattern } from './geometry.js';
import { createViewport, renderThumbnail } from './viewport.js';
import { icon, applyIcons } from './icons.js';
import { strings } from './strings.js';
import { state, full, linesFor, overlayLayers, togglePattern, designURL, persist } from './design.js';
import { analyzeNet } from './metrics.js';
import { drawNet, highlightCrease, markMeasurement } from './net.js';

const byId = (id) => document.getElementById(id);
const t = (key) => strings[state.lang][key];
const $all = (query, root = document) => [...root.querySelectorAll(query)];
const code = (id) => id[0] + '<sub>' + id[1] + '</sub>' + (id.endsWith('m') ? '<sup>m</sup>' : '');
const family = (p, extended = false) => t(extended && p.id[0] === 'Q' ? 'Qfull' : p.id[0]);
function tooltip(el, label) {
  el.setAttribute('aria-label', label);
  el.title = label;
}
const viewport = createViewport(byId('model-canvas'));
let animation = 0,
  noticeTimer,
  guide = false;
function toast(message) {
  clearTimeout(noticeTimer);
  byId('notice').textContent = message;
  noticeTimer = setTimeout(() => (byId('notice').textContent = ''), 2600);
}
function languageUI() {
  document.documentElement.lang = state.lang;
  $all('[data-i]').forEach((el) => (el.textContent = t(el.dataset.i)));
  $all('[data-tip]').forEach((el) => tooltip(el, t(el.dataset.tip)));
  for (const lang of ['ja', 'en'])
    byId('lang-' + lang).setAttribute('aria-pressed', String(lang === state.lang));
  tooltip(byId('model-canvas'), t('canvasHelp'));
  byId('model-canvas').setAttribute('role', 'img');
  drawPlay();
}
function drawPlay() {
  byId('play').innerHTML = icon(animation ? 'pause' : 'play');
  tooltip(byId('play'), t(animation ? 'pause' : 'play'));
}
function setProgress(value) {
  state.progress = Math.max(0, Math.min(100, Math.round(value)));
  byId('progress').value = state.progress;
  byId('progress-value').textContent = state.progress + '%';
  byId('progress').style.setProperty('--progress', state.progress + '%');
  viewport.setProgress(state.progress / 100);
}
function stop() {
  cancelAnimationFrame(animation);
  animation = 0;
  drawPlay();
}
function play() {
  if (animation) {
    stop();
    persist();
    return;
  }
  if (state.progress >= 100) setProgress(0);
  const start = performance.now(),
    initial = state.progress;
  function advance(time) {
    const percentage = initial + (time - start) / 38;
    if (Math.round(percentage) !== state.progress) setProgress(percentage);
    if (percentage >= 100) {
      stop();
      persist();
    } else animation = requestAnimationFrame(advance);
  }
  animation = requestAnimationFrame(advance);
  drawPlay();
}
let mirrors = state.model.id.endsWith('m');
function drawPatternList() {
  byId('mode-original').setAttribute('aria-pressed', String(!mirrors));
  byId('mode-mirror').setAttribute('aria-pressed', String(mirrors));
  const list = byId('pattern-list');
  list.replaceChildren(
    ...ALL_PATTERNS.filter((p) => p.id.endsWith('m') === mirrors).map((p) => {
      const active = state.patterns.includes(p),
        btn = document.createElement('button');
      btn.className = 'pattern-card';
      btn.dataset.pattern = p.id;
      btn.setAttribute('aria-pressed', String(active));
      tooltip(btn, p.id + ' · ' + family(p, true));
      btn.innerHTML =
        '<svg aria-hidden="true" class="sample"></svg><span class="math-code">' +
        code(p.id) +
        '</span><span class="family">' +
        family(p) +
        '</span>' +
        (active ? '<span class="selected-mark">' + icon('check') + '</span>' : '');
      renderThumbnail(btn.querySelector('svg'), p, active);
      btn.onclick = () => {
        togglePattern(p);
        refreshDesign();
        byId('pattern-list')
          .querySelector('[data-pattern=' + p.id + ']')
          .focus({ preventScroll: true });
      };
      return btn;
    }),
  );
}
function keepPatternVisible() {
  const list = byId('pattern-list'),
    item = list.querySelector('[data-pattern=' + state.model.id + ']');
  if (!item || !list.clientWidth) return;
  const bounds = list.getBoundingClientRect(),
    rect = item.getBoundingClientRect();
  if (rect.right > bounds.right || rect.left < bounds.left)
    list.scrollLeft += rect.left - bounds.left - (bounds.width - rect.width) / 2;
}
function refreshDesign() {
  stop();
  syncModel();
  persist();
}
function chooseTarget(id) {
  const p = getPattern(id);
  if (!state.patterns.includes(p)) return;
  state.model = p;
  refreshDesign();
}
let metricKey = null;
function stats() {
  const lines = state.patterns.flatMap(linesFor);
  const metrics = analyzeNet(lines, state.edge);
  const locale = state.lang === 'ja' ? 'ja-JP' : 'en-US';
  const fmt = (n, precision = 1) => n.toLocaleString(locale, { maximumFractionDigits: precision });
  byId('selected-count').textContent =
    state.patterns.length + ' ' + (state.lang === 'ja' ? '選択中' : 'selected');
  byId('panel-count').textContent = metrics.panelCount + ' ' + t('panels');
  const specs = [
    ['angle', metrics.minAngleDeg, '°'],
    ['area', metrics.minAreaMm2, ' mm²'],
    ['spacing', metrics.minSpacingMm, ' mm'],
  ];
  byId('metric-values').innerHTML = specs
    .map(
      ([key, value, unit]) =>
        '<button class="metric ' +
        (metrics.warnings.includes(key) ? 'warn' : '') +
        '" data-metric="' +
        key +
        '" aria-pressed="' +
        (metricKey === key) +
        '"><span>' +
        t(key) +
        '</span><strong>' +
        fmt(value, 2) +
        '<small>' +
        unit +
        '</small></strong></button>',
    )
    .join('');
  markMeasurement(byId('net-preview'), metricKey, metrics);
  const limits = metrics.limits;
  byId('metric-note').textContent = metrics.warnings.length
    ? t('featureWarning') +
      metrics.warnings
        .map(
          (k) =>
            t(k) +
            ' < ' +
            (k === 'angle'
              ? limits.angleDeg + '°'
              : k === 'area'
                ? limits.areaMm2 + ' mm²'
                : limits.spacingMm + ' mm'),
        )
        .join(' · ')
    : '';
  byId('metric-note').hidden = !metrics.warnings.length;
  byId('design-size').value = state.edge;
  byId('pattern-clear').textContent = t(state.patterns.length ? 'clear' : 'all');
}
function drawTargets() {
  const select = byId('target-select');
  select.innerHTML = state.patterns.length
    ? state.patterns
        .map(
          (p) =>
            '<option value="' +
            p.id +
            '" ' +
            (p === state.model ? 'selected' : '') +
            '>' +
            p.id +
            ' · ' +
            family(p, true) +
            '</option>',
        )
        .join('')
    : '<option>' + t('blank') + '</option>';
  select.disabled = state.patterns.length < 2;
  tooltip(select, t('target'));
  byId('edit-open').disabled = !state.patterns.length;
  byId('play').disabled = !state.selected.size;
  byId('progress').disabled = !state.selected.size;
}
function drawPreview() {
  const { model, selected, overlays, edge, lang } = state;
  drawNet(byId('net-preview'), model, selected, overlayLayers(), edge, lang);
  const legend = byId('net-preview').closest('.net-card').querySelector('.legend');
  legend.innerHTML =
    '<span><b class="cut-line"></b>' +
    t('cut') +
    '</span><span><b class="crease-line"></b>' +
    t('crease') +
    '</span>' +
    (overlays.length ? '<span><b class="overlay-line"></b>' + t('extra') + '</span>' : '');
  byId('model-code').innerHTML = code(model.id);
  byId('model-name').textContent = !state.patterns.length
    ? t('blank')
    : full()
      ? family(model, true)
      : t('partial');
  byId('model-code').hidden = !state.patterns.length;
  drawTargets();
  stats();
  document.querySelector('.range-ends span:last-child').textContent = t(full() ? 'folded' : 'partial');
}
function syncModel() {
  viewport.setModel(state.model, state.progress / 100, state.selected);
  viewport.setOverlays(overlayLayers().flatMap((p) => p.creases));
  setProgress(state.progress);
  drawPreview();
  drawPatternList();
}
function dialogHead(id, title) {
  return (
    '<div class="dialog-heading"><h2 id="' +
    id +
    '">' +
    title +
    '</h2><button class="icon-btn" data-close aria-label="' +
    t('close') +
    '">' +
    icon('close') +
    '</button></div>'
  );
}
function showDialog(el) {
  stop();
  el.showModal();
}
let showSeams = false;
function drawEditor() {
  const content = byId('editor-content'),
    model = state.model,
    active = state.selected;
  const focused = document.activeElement?.dataset;
  const previous =
    focused &&
    (focused.crease
      ? '[data-crease="' + focused.crease + '"]'
      : focused.overlay
        ? '[data-overlay="' + focused.overlay + '"]'
        : focused.action
          ? '[data-action="' + focused.action + '"]'
          : null);
  content.innerHTML =
    dialogHead('editor-title', t('edit')) +
    '<div class="editor-layout"><div class="editor-sheet"><svg id="editor-net"></svg></div><div class="editor-options">' +
    '<div class="form-unit"><label class="subhead" for="edit-target">' +
    t('editLayer') +
    '</label><select id="edit-target">' +
    state.patterns
      .map(
        (p) =>
          '<option value="' +
          p.id +
          '" ' +
          (p === model ? 'selected' : '') +
          '>' +
          p.id +
          ' · ' +
          family(p, true) +
          '</option>',
      )
      .join('') +
    '</select></div>' +
    '<div class="form-unit"><div class="subhead"><span>' +
    t('choose') +
    '</span><button class="text-btn" data-action="toggle-all">' +
    t(full() ? 'clear' : 'all') +
    '</button></div>' +
    '<p class="hint">' +
    t('chooseHint') +
    '</p><div class="chip-row">' +
    model.creases
      .map(
        (c) =>
          '<button class="chip" data-crease="' +
          c.id +
          '" aria-pressed="' +
          active.has(c.id) +
          '">' +
          c.id +
          '</button>',
      )
      .join('') +
    '</div></div>' +
    '<label class="seam-switch"><input id="show-seams" type="checkbox" ' +
    (showSeams ? 'checked' : '') +
    '>' +
    t('showSeams') +
    '</label></div></div><div class="bottom-action"><button class="btn primary" data-close>' +
    t('done') +
    '</button></div>';
  byId('edit-target').onchange = (e) => {
    chooseTarget(e.target.value);
    drawEditor();
    byId('edit-target').focus();
  };
  byId('show-seams').onchange = (e) => {
    showSeams = e.target.checked;
    drawEditor();
    byId('show-seams').focus();
  };
  drawNet(
    byId('editor-net'),
    model,
    active,
    overlayLayers(),
    state.edge,
    state.lang,
    toggleCrease,
    showSeams,
  );
  const highlight = (id) =>
    highlightCrease(
      byId('editor-net'),
      model.creases.find((c) => c.id === id),
    );
  for (const chip of content.querySelectorAll('[data-crease]')) {
    chip.onmouseenter = chip.onfocus = () => highlight(chip.dataset.crease);
    chip.onmouseleave = chip.onblur = () => highlight(null);
  }
  if (previous) content.querySelector(previous)?.focus({ preventScroll: true });
}
function changedFolds() {
  viewport.setCreases(state.selected);
  if (!state.selected.size) setProgress(0);
  drawPreview();
  drawEditor();
  persist();
}
function toggleCrease(id) {
  if (state.selected.has(id)) state.selected.delete(id);
  else state.selected.add(id);
  changedFolds();
}
byId('editor-content').addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (!btn) return;
  if (btn.dataset.crease) toggleCrease(btn.dataset.crease);
  if (btn.dataset.action === 'toggle-all') {
    state.selected = full() ? new Set() : new Set(state.model.creases.map((c) => c.id));
    changedFolds();
  }
});
const formats = [
  ['pdf', 'paper'],
  ['solid', 'cube'],
  ['kit', 'layers'],
];
function exportNote() {
  if (state.format === 'solid')
    return !full()
      ? t('solidIncomplete')
      : t('solidNote') + (state.model.id.startsWith('Q') ? ' ' + t('qNote') : '');
  if (state.format === 'kit') return t(state.method + 'Note');
  if (!state.patterns.length)
    return state.lang === 'ja'
      ? '折り線を入れず、十字形の外周を原寸で出力します。'
      : 'Export the uncreased outline at actual size.';
  return t(state.overlays.length ? 'compositePDF' : full() ? 'pdfNote' : 'partialPDF');
}
function drawExport() {
  const disabled = state.format === 'solid' && !full();
  const modelIds = state.patterns.length ? state.patterns.map((p) => p.id).join(' + ') : t('blank');
  byId('export-content').innerHTML =
    dialogHead('export-title', t('design')) +
    '<div class="subhead">' +
    modelIds +
    '</div><div class="format-group">' +
    formats
      .map(
        ([f, i]) =>
          '<button class="format-choice" data-format="' +
          f +
          '" aria-pressed="' +
          (state.format === f) +
          '">' +
          icon(i) +
          '<span><strong>' +
          t(f) +
          '</strong><small>' +
          t(f + 'Sub') +
          '</small></span><span class="radio-dot"></span></button>',
      )
      .join('') +
    '</div>' +
    '<div class="form-unit"><div class="field-row"><label for="edge-mm">' +
    t('unit') +
    '</label><div><input type="number" min="20" max="60" step="0.1" id="edge-mm" required value="' +
    state.edge +
    '"><span class="unit">mm</span></div></div>' +
    (state.format === 'kit'
      ? '<div class="field-row"><label for="joint-type">' +
        t('method') +
        '</label><select id="joint-type">' +
        ['tape', 'magnet', 'snap']
          .map(
            (m) =>
              '<option value="' +
              m +
              '" ' +
              (m === state.method ? 'selected' : '') +
              '>' +
              t(m) +
              '</option>',
          )
          .join('') +
        '</select></div>'
      : '') +
    (state.format !== 'solid'
      ? '<div class="field-row"><label for="paper-size">' +
        t('paper') +
        '</label><select id="paper-size"><option>A4</option><option' +
        (state.paper === 'Letter' ? ' selected' : '') +
        '>Letter</option></select></div>'
      : '') +
    '</div>' +
    '<p class="hint">' +
    exportNote() +
    '</p>' +
    (state.format === 'kit' && state.method !== 'tape'
      ? '<button class="text-btn coupon" id="coupon-download">' + t('coupon') + '</button>'
      : '') +
    '<p id="export-error" class="hint error" role="alert"></p><div class="bottom-action"><button id="download-file" class="btn primary"' +
    (disabled ? ' disabled' : '') +
    '>' +
    icon('download') +
    t('download') +
    ' ' +
    (state.format === 'pdf' ? 'PDF' : state.format === 'kit' ? 'ZIP' : 'STL') +
    '</button></div>';
  byId('edge-mm').addEventListener('input', (e) => {
    if (e.target.validity.valid) {
      state.edge = Number(e.target.value);
      drawPreview();
      persist();
    }
    byId('export-error').textContent = '';
  });
  byId('paper-size')?.addEventListener('change', (e) => {
    state.paper = e.target.value;
    persist();
  });
  if (byId('joint-type'))
    byId('joint-type').onchange = (e) => {
      state.method = e.target.value;
      drawExport();
      byId('joint-type').focus();
    };
  if (byId('coupon-download')) byId('coupon-download').onclick = () => download(true);
  byId('download-file').onclick = () => download(false);
}
let exportWorker;
byId('export-dialog').addEventListener('close', () => {
  exportWorker?.terminate();
  exportWorker = undefined;
});
async function download(couponOnly = false) {
  if (exportWorker) return;
  const input = byId('edge-mm');
  if (!couponOnly && !input.validity.valid) {
    byId('export-error').textContent = t('invalidSize');
    input.focus();
    return;
  }
  const { model, patterns, edge: edgeMm, lang, paper, format, method } = state;
  const request = {
    target: model.id,
    patterns: patterns.map((p) => p.id),
    selections: Object.fromEntries(patterns.map((p) => [p.id, linesFor(p).map((c) => c.id)])),
    format,
    method,
    edgeMm,
    lang,
    paper,
    couponOnly,
  };
  const failed = (error) => {
    exportWorker?.terminate();
    exportWorker = undefined;
    drawExport();
    byId('export-error').textContent = t(
      error.name === 'RangeError'
        ? format === 'kit' && method !== 'tape'
          ? 'kitFit'
          : 'tooSmall'
        : 'exportError',
    );
  };
  try {
    exportWorker = new Worker(new URL('./export-worker.js', import.meta.url), { type: 'module' });
    const exportButton = byId(couponOnly ? 'coupon-download' : 'download-file');
    exportButton.textContent = lang === 'ja' ? '準備中…' : 'Preparing…';
    byId('export-content')
      .querySelectorAll('input,select,button:not([data-close])')
      .forEach((el) => (el.disabled = true));
    exportWorker.onerror = failed;
    exportWorker.onmessage = async ({ data }) => {
      if (data.error) {
        failed({ name: data.error });
        return;
      }
      const { saveBlob } = await import('./fabrication.js');
      if (!byId('export-dialog').open) return;
      saveBlob(data.blob, data.filename);
      byId('export-dialog').close();
      toast(t('exported'));
    };
    exportWorker.postMessage(request);
  } catch (error) {
    failed(error);
  }
}
let guideMethod = 'paper';
async function guidePage() {
  const { renderGuide } = await import('./guide-view.js');
  renderGuide(
    byId('guide-page'),
    state.lang,
    guideMethod,
    (method) => {
      guideMethod = method;
      guidePage().then(() =>
        byId('guide-page').querySelector('[aria-pressed=true]')?.focus({ preventScroll: true }),
      );
    },
    () => navigate(false),
  );
}
function navigate(showGuide) {
  stop();
  guide = showGuide;
  byId('studio').hidden = guide;
  byId('guide-page').hidden = !guide;
  byId('studio-nav').classList.toggle('active', !guide);
  byId('guide-nav').classList.toggle('active', guide);
  byId('studio-nav').setAttribute('aria-current', guide ? 'false' : 'page');
  byId('guide-nav').setAttribute('aria-current', guide ? 'page' : 'false');
  if (guide) guidePage();
  else keepPatternVisible();
  window.scrollTo({ top: 0 });
}
function switchLanguage(lang) {
  if (state.lang === lang) return;
  state.lang = lang;
  languageUI();
  drawPreview();
  drawPatternList();
  if (guide) guidePage();
  persist();
}
async function share() {
  const url = designURL();
  try {
    if (!navigator.clipboard) throw new Error('Clipboard unavailable');
    await navigator.clipboard.writeText(url);
    toast(t('copied'));
  } catch {
    byId('share-content').innerHTML =
      dialogHead('share-title', t('shareTitle')) +
      '<p class="hint">' +
      t('shareHelp') +
      '</p><input class="share-url" id="share-url" readonly aria-label="URL">';
    byId('share-url').value = url;
    showDialog(byId('share-dialog'));
    byId('share-url').focus();
    byId('share-url').select();
  }
}
const largeView = document.createElement('dialog');
largeView.className = 'large-view';
largeView.setAttribute('aria-labelledby', 'sim-title');
document.body.append(largeView);
const card = byId('sim-card'),
  bench = card.parentElement;
function expandView() {
  if (largeView.open) {
    largeView.close();
    return;
  }
  largeView.append(card);
  largeView.showModal();
  byId('expand').innerHTML = icon('collapse');
  tooltip(byId('expand'), t('collapse'));
}
largeView.addEventListener('close', () => {
  bench.prepend(card);
  byId('expand').innerHTML = icon('expand');
  tooltip(byId('expand'), t('expand'));
  byId('expand').focus();
});
byId('metric-values').onclick = (e) => {
  const chip = e.target.closest('[data-metric]');
  if (!chip) return;
  metricKey = metricKey === chip.dataset.metric ? null : chip.dataset.metric;
  stats();
  byId('metric-values')
    .querySelector('[data-metric=' + chip.dataset.metric + ']')
    .focus({ preventScroll: true });
};
byId('mode-original').onclick = () => {
  mirrors = false;
  drawPatternList();
};
byId('mode-mirror').onclick = () => {
  mirrors = true;
  drawPatternList();
};
byId('target-select').onchange = (e) => chooseTarget(e.target.value);
byId('pattern-clear').onclick = () => {
  if (state.patterns.length) {
    state.patterns = [];
    state.progress = 0;
  } else {
    state.patterns = [...ALL_PATTERNS];
    state.progress = 100;
  }
  refreshDesign();
};
byId('design-size').onchange = (e) => {
  if (e.target.validity.valid) {
    state.edge = Number(e.target.value);
    drawPreview();
    persist();
  } else {
    e.target.reportValidity();
    e.target.value = state.edge;
  }
};
byId('lang-ja').onclick = () => switchLanguage('ja');
byId('lang-en').onclick = () => switchLanguage('en');
byId('studio-nav').onclick = () => navigate(false);
byId('guide-nav').onclick = () => navigate(true);
byId('brand').onclick = (e) => {
  if (!e.ctrlKey && !e.metaKey) {
    e.preventDefault();
    navigate(false);
  }
};
byId('play').onclick = play;
byId('progress').oninput = (e) => {
  stop();
  setProgress(Number(e.target.value));
};
byId('progress').onchange = persist;
byId('share').onclick = share;
byId('expand').onclick = expandView;
byId('zoom-in').onclick = () => viewport.zoomBy(1.2);
byId('zoom-out').onclick = () => viewport.zoomBy(1 / 1.2);
byId('reset-view').onclick = viewport.resetView;

byId('edit-open').onclick = () => {
  drawEditor();
  showDialog(byId('editor-dialog'));
};
byId('export-open').onclick = () => {
  drawExport();
  showDialog(byId('export-dialog'));
};
byId('export-content').onclick = (e) => {
  const btn = e.target.closest('[data-format]');
  if (btn) {
    state.format = btn.dataset.format;
    drawExport();
    byId('export-content').querySelector('[aria-pressed=true]').focus();
  }
};
document.addEventListener('click', (e) => {
  const close = e.target.closest('[data-close]');
  if (close) close.closest('dialog').close();
  if (e.target instanceof HTMLDialogElement && e.target.open) {
    const box = e.target.getBoundingClientRect();
    if (e.clientX < box.left || e.clientX > box.right || e.clientY < box.top || e.clientY > box.bottom)
      e.target.close();
  }
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden && animation) {
    stop();
    persist();
  }
});
applyIcons();
languageUI();
syncModel();
keepPatternVisible();
byId('studio-nav').setAttribute('aria-current', 'page');
