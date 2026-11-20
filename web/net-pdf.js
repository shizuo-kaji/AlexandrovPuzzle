// The page uses physical millimetres. Keep all coordinates and line widths in
// that unit until the PDF boundary so changing paper never rescales the net.
import { OUTLINE } from './geometry.js';
import { fontData, glyphs } from './fonts/japanese.js';

const MM = 72 / 25.4;
const CUT = [0.1, 0.15, 0.16];
const FOLD = [0.1, 0.39, 0.36];
const MUTED = [0.25, 0.29, 0.3];
const strings = {
  ja: {
    title: '展開図',
    cut: '切り線',
    fold: '折り線',
    join: '同じ文字の辺をテープで合わせる',
    overlay: '作る形に合わせて折り線を選ぶ',
    actual: '原寸で印刷',
    edge: '正方形の一辺',
    count: '選択した折り線',
    partial: '完成形を作るにはすべての折り線を選択してください。',
  },
  en: {
    title: 'Net template',
    cut: 'Cut',
    fold: 'Fold',
    join: 'Tape together edges with matching letters.',
    overlay: 'Use the crease set for the shape you want to fold.',
    actual: 'Print at actual size',
    edge: 'Square edge',
    count: 'Selected creases',
    partial: 'Select every crease to assemble the closed model.',
  },
};
const num = (value) => Number(value.toFixed(3)).toString();
const u16 = (value) => value.toString(16).padStart(4, '0');
const ascii = (str) => '(' + str.replace(/[()\\]/g, '\\$&') + ')';

class PDF {
  objects = [];
  object(body) {
    this.objects.push(body);
    return this.objects.length;
  }
  stream(data, attributes = '') {
    return this.object(`<< /Length ${data.length} ${attributes} >>\nstream\n${data}\nendstream`);
  }
  japaneseFont() {
    const name = 'APZLJP+AlexandrovPDFJP';
    const file = this.stream(
      fontData + '>',
      '/Subtype /CIDFontType0C /Filter [/ASCIIHexDecode /FlateDecode]',
    );
    const desc = this.object(
      `<< /Type /FontDescriptor /FontName /${name} /Flags 4 /FontBBox [-1002 -1048 2928 1808] /ItalicAngle 0 /Ascent 880 /Descent -120 /CapHeight 733 /StemV 80 /FontFile3 ${file} 0 R >>`,
    );
    const info = '<< /Registry (Adobe) /Ordering (Japan1) /Supplement 7 >>';
    const cid = this.object(
      `<< /Type /Font /Subtype /CIDFontType0 /BaseFont /${name} /CIDSystemInfo ${info} /FontDescriptor ${desc} 0 R /DW 1000 >>`,
    );
    const mappings = Object.entries(glyphs).map(
      ([char, id]) => `<${u16(id)}> <${u16(char.codePointAt(0))}>`,
    );
    const uni = this.stream(
      `/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n/CMapName /AlexandrovPuzzleUnicode def\n/CMapType 2 def\n1 begincodespacerange\n<0000> <ffff>\nendcodespacerange\n${mappings.length} beginbfchar\n${mappings.join('\n')}\nendbfchar\nendcmap\nCMapName currentdict /CMap defineresource pop\nend end`,
    );
    return this.object(
      `<< /Type /Font /Subtype /Type0 /BaseFont /${name} /Encoding /Identity-H /DescendantFonts [${cid} 0 R] /ToUnicode ${uni} 0 R >>`,
    );
  }
  finish(root) {
    let body = '%PDF-1.5\n',
      xref = '0000000000 65535 f \n';
    for (const [index, obj] of this.objects.entries()) {
      xref += `${String(body.length).padStart(10, '0')} 00000 n \n`;
      body += `${index + 1} 0 obj\n${obj}\nendobj\n`;
    }
    const offset = body.length,
      size = this.objects.length + 1;
    body += `xref\n0 ${size}\n${xref}trailer\n<< /Size ${size} /Root ${root} 0 R >>\nstartxref\n${offset}\n%%EOF\n`;
    return new Blob([body], { type: 'application/pdf' });
  }
}

function drawNetPage(model, { edgeMm, selected, japanese, w, h, pageIndex, count }) {
  const ids = new Set(selected ?? model.creases.map((c) => c.id));
  const active = model.creases.filter((c) => ids.has(c.id));
  const t = strings[japanese ? 'ja' : 'en'];
  const ox = (w - edgeMm * 3) / 2,
    oy = 18 + (h - 37 - edgeMm * 4) / 2;
  const world = (p) => [ox + edgeMm * p[0], oy + edgeMm * p[1]];
  const pagePoint = ([x, y]) => num(x * MM) + ' ' + num((h - y) * MM);
  const commands = [];
  function line(points, width, color, dashes = [], close = false) {
    const path = points.map((p, i) => pagePoint(p) + (i ? ' l' : ' m')).join(' ');
    commands.push(
      `${color.join(' ')} RG ${num(width * MM)} w [${dashes.map((x) => num(x * MM)).join(' ')}] 0 d ${path} ${close ? 'h ' : ''}S`,
    );
  }
  function textAt(str, x, y, pt = 9, color = CUT, bold = false) {
    const isJP = /[^\x20-\x7e]/.test(str);
    const label = isJP ? '<' + [...str].map((c) => u16(glyphs[c])).join('') + '>' : ascii(str);
    commands.push(
      `BT /${isJP ? 'J' : bold ? 'B' : 'R'} ${pt} Tf ${color.join(' ')} rg 1 0 0 1 ${pagePoint([x, y])} Tm ${label} Tj ET`,
    );
  }
  textAt('Alexandrov Puzzle', 9, 10, 16, CUT, true);
  textAt(t.title, 65, 10, 12);
  textAt(model.id, 9, 15.2, 9, MUTED);
  const rulerX = w - 30;
  line(
    [
      [rulerX, 12],
      [rulerX + 20, 12],
    ],
    0.28,
    CUT,
  );
  for (const x of [0, 10, 20])
    line(
      [
        [rulerX + x, 10.8],
        [rulerX + x, 13.2],
      ],
      0.23,
      CUT,
    );
  textAt('20 mm', rulerX + 5, 9.5, 8.5, MUTED);
  for (const crease of active) line([world(crease.a), world(crease.b)], 0.26, FOLD, [1.8, 1.15]);
  line(OUTLINE.map(world), 0.42, CUT, [], true);
  for (const { a, b, label } of model.seams ?? []) {
    const dx = b[0] - a[0],
      dy = b[1] - a[1],
      length = Math.hypot(dx, dy);
    const normal = [-dy / length, dx / length],
      mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    // Seams follow the counterclockwise outline, so the left normal points in.
    const pos = world(mid).map((v, i) => v + 2.1 * normal[i]);
    textAt(label, pos[0] - 0.95, pos[1] + 0.85, 7.5, MUTED, true);
  }
  if (count > 1) textAt(pageIndex + 1 + ' / ' + count, w - 18, h - 6.8, 8, MUTED);
  const meta = h - 13;
  line(
    [
      [9, meta],
      [17, meta],
    ],
    0.42,
    CUT,
  );
  textAt(t.cut, 19, meta + 1.1, 9);
  line(
    [
      [36, meta],
      [44, meta],
    ],
    0.26,
    FOLD,
    [1.8, 1.15],
  );
  textAt(t.fold, 46, meta + 1.1, 9);
  textAt(model.seams?.length ? t.join : t.overlay, 66, meta + 1.1, 9);
  textAt(t.actual, 9, h - 6.8, 8, MUTED);
  textAt('100%', 34, h - 6.8, 8, MUTED, true);
  textAt(t.edge, 53, h - 6.8, 8, MUTED);
  textAt(edgeMm + ' mm', 78, h - 6.8, 8, MUTED, true);
  textAt(t.count, 100, h - 6.8, 8, MUTED);
  textAt(active.length + ' / ' + model.creases.length, 129, h - 6.8, 8, MUTED, true);
  if (model.seams?.length && active.length < model.creases.length) {
    textAt(t.partial, 9, 18.5, 7.5, MUTED);
  }
  return commands.join('\n');
}

export function createNetPDF(
  model,
  { edgeMm = 40, selected, lang = 'en', paper = 'A4', targets = [], targetSelected, selections = {} } = {},
) {
  edgeMm = Number(edgeMm);
  if (!(edgeMm >= 20 && edgeMm <= 60)) throw new RangeError('Square edge must be between 20 and 60 mm');
  const japanese = lang === 'ja';
  const [w, h] = paper.toLowerCase() === 'letter' ? [215.9, 279.4] : [210, 297];
  const sheets = [
    { model, selected },
    ...targets.map((target, index) => ({
      model: target,
      selected: selections[target.id] ?? (index === 0 ? targetSelected : undefined),
    })),
  ];
  const pdf = new PDF(),
    root = pdf.object(null),
    pages = pdf.object(null);
  const reg = pdf.object('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>');
  const bold = pdf.object('<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>');
  const jp = japanese ? `/J ${pdf.japaneseFont()} 0 R` : '';
  const refs = sheets.map((sheet, pageIndex) => {
    const content = pdf.stream(
      drawNetPage(sheet.model, {
        edgeMm,
        selected: sheet.selected,
        japanese,
        w,
        h,
        pageIndex,
        count: sheets.length,
      }),
    );
    const page = pdf.object(
      `<< /Type /Page /Parent ${pages} 0 R /MediaBox [0 0 ${num(w * MM)} ${num(h * MM)}] /Resources << /Font << /R ${reg} 0 R /B ${bold} 0 R ${jp} >> >> /Contents ${content} 0 R >>`,
    );
    return page + ' 0 R';
  });
  pdf.objects[root - 1] =
    `<< /Type /Catalog /Pages ${pages} 0 R /ViewerPreferences << /PrintScaling /None >> >>`;
  pdf.objects[pages - 1] = `<< /Type /Pages /Kids [${refs.join(' ')}] /Count ${refs.length} >>`;
  return pdf.finish(root);
}
