import { getPattern } from './geometry.js';
import { combinedNet } from './net.js';
import { createNetPDF, createSolidSTL } from './fabrication.js';

export async function exportDesign(request) {
  const { format, method, edgeMm, lang, paper, couponOnly, selections } = request;
  const model = getPattern(request.target);
  const ordered = request.patterns.map(getPattern);
  const fullTargets = ordered.filter((p) => selections[p.id].length === p.creases.length);
  const selected = new Set(selections[model.id] || []);
  const extras = ordered
    .filter((p) => p !== model)
    .map((p) => ({ ...p, creases: p.creases.filter((c) => selections[p.id].includes(c.id)) }));
  const net = ordered.length
    ? combinedNet(model, selected, extras)
    : { id: 'blank', creases: [], seams: [] };
  const options = {
    edgeMm,
    lang,
    paper,
    selected: extras.length ? undefined : selected,
    targets: extras.length ? ordered : undefined,
    selections,
    completeTargets: fullTargets,
  };
  let blob;
  if (format === 'pdf') blob = createNetPDF(net, options);
  else if (format === 'solid') {
    if (!fullTargets.includes(model)) throw new Error('Incomplete target');
    blob = createSolidSTL(model, { edgeMm, extraCreases: extras.flatMap((p) => p.creases) });
  } else {
    const { makeKit } = await import('./kits.js');
    blob = await makeKit(method, net, options, couponOnly);
  }
  const extension = format === 'pdf' ? 'pdf' : format === 'kit' ? 'zip' : 'stl';
  const label = couponOnly
    ? 'sample-' + method
    : net.id + '-' + (format === 'kit' ? method : format) + '-' + edgeMm + 'mm';
  return { blob, filename: 'AlexandrovPuzzle-' + label + '.' + extension };
}
