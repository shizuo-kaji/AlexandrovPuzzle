import { createNetPDF, createKitSTL } from './fabrication.js';
import { zipFiles } from './archive.js';
import { instructionsText } from './build-guide.js';

export async function makeKit(method, model, options, couponOnly = false) {
  const files = {};
  const details = {
    pattern: model.id,
    patterns: options.targets?.map((p) => p.id) || [model.id],
    edgeMm: options.edgeMm,
    paper: options.paper,
    coupon: couponOnly,
    completeTargets: options.completeTargets,
  };
  if (!couponOnly) files['template.pdf'] = createNetPDF(model, options);
  if (method === 'tape') {
    files['panels.stl'] = createKitSTL(model, options);
  } else if (method === 'magnet') {
    const { createMagnetKit, createMagnetCoupon } = await import('./magnet-kit.js');
    if (!couponOnly) {
      const result = createMagnetKit(model, { ...options, targets: options.completeTargets });
      files['panels-magnet.stl'] = result.panels;
      if (result.assemblySVG) files['magnet-placement.svg'] = result.assemblySVG;
      Object.assign(details, {
        panelCount: result.panelCount,
        pocketCount: result.pocketCount,
        bridgeCount: result.bridgeCount,
        magnetCount: result.magnetCount,
      });
    }
    files['sample-magnet.stl'] = createMagnetCoupon().panels;
  } else {
    const { createSnapKit, createSnapCoupon } = await import('./snap-kit.js');
    if (!couponOnly) {
      const result = createSnapKit(model, { ...options, seamPatterns: options.completeTargets });
      files['panels-rigid.stl'] = result.panels;
      if (result.assemblySVG) files['snap-placement.svg'] = result.assemblySVG;
      if (result.links?.size > 84) files['links-TPU.stl'] = result.links;
      Object.assign(details, {
        panelCount: result.panelCount,
        hingeLinks: result.hingeLinks,
        seamLinks: result.seamLinks,
      });
    }
    const sample = createSnapCoupon();
    files['sample-rigid.stl'] = sample.panels;
    files['sample-TPU.stl'] = sample.links;
  }
  details.files = Object.keys(files);
  for (const lang of ['ja', 'en']) {
    const instructions = instructionsText(method, lang, details);
    files['BUILD-' + lang + '.txt'] = instructions;
  }
  return zipFiles(files);
}
