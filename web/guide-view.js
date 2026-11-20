import { guideSection } from './build-guide.js';
import { icon } from './icons.js';
import { strings } from './strings.js';

const methods = ['paper', 'tape', 'magnet', 'snap', 'solid'];
function diagram(method, lang) {
  const ja = lang === 'ja',
    ink = '#443c33',
    clay = '#c36946',
    plate = '#dfc2a2',
    flex = '#4e756c';
  const title = ja ? 'ヒンジの断面模式図' : 'Hinge cross section';
  const labels = {
    tape: ja
      ? ['薄いテープ', '滑らかな面', '3.2 mm のすき間']
      : ['Thin tape', 'Smooth outer face', '3.2 mm nominal gap'],
    magnet: ja
      ? ['磁石入りの布', '磁石を埋めたパネル', '4.4 mm のすき間']
      : ['Sewn magnetic bridge', 'Recessed panel magnet', '4.4 mm gap'],
    snap: ja
      ? ['TPUリンク', 'スナップボタン', '硬質パネル']
      : ['Flexible TPU link', 'Snap stud', 'Rigid panel'],
  };
  if (!labels[method]) return '';
  const [top, detail, bottom] = labels[method];
  let shapes = '';
  if (method === 'tape')
    shapes =
      '<path d="M34 85h151v11l-8 12H34zM202 85h150v23H210l-8-12z" fill="' +
      plate +
      '" stroke="' +
      ink +
      '"/><path d="M99 80q93 3 186 0v5q-93 2-186 0z" fill="' +
      flex +
      '"/>';
  if (method === 'magnet')
    shapes =
      '<path d="M34 84h82v12h25V84h42v25H34z M205 84h43v12h25V84h78v25H205z" fill="' +
      plate +
      '" stroke="' +
      ink +
      '"/><path d="M117 85h23v10h-23zM249 85h23v10h-23z" fill="' +
      clay +
      '"/><path d="M114 81q3-17 16-16t16 8q51 4 98 0q3-10 16-9t17 17q-80 4-163 0z" fill="' +
      flex +
      '"/><path d="M118 72h23v8h-23zM249 72h23v8h-23z" fill="#d3b18e"/>';
  if (method === 'snap')
    shapes =
      '<path d="M35 92h147v22H35zM204 92h148v22H204zM121 92V72h-5v-5l5-6h15l5 6v5h-5v20zM250 92V72h-5v-5l5-6h15l5 6v5h-5v20z" fill="' +
      plate +
      '" stroke="' +
      ink +
      '"/><path d="M105 76h15v11h-15zM137 76h112v11H137zM266 76h15v11h-15z" fill="' +
      flex +
      '"/>';
  const text = (x, y, label) =>
    '<text x="' + x + '" y="' + y + '" font-size="12" fill="' + ink + '">' + label + '</text>';
  return (
    '<svg class="guide-diagram" viewBox="0 0 392 155" role="img" aria-label="' +
    title +
    '"><title>' +
    title +
    '</title>' +
    shapes +
    '<g fill="none" stroke="#847767" stroke-width="1"><path d="m153 73 15-22h30"/><path d="m' +
    (method === 'tape' ? '93 88' : '130 70') +
    ' -19-23H33"/><path d="m' +
    (method === 'snap' ? '285 107' : '194 102') +
    ' 15 25h33"/></g>' +
    text(167, 44, top) +
    text(32, 39, detail) +
    text(method === 'snap' ? 231 : 184, 145, bottom) +
    '</svg>'
  );
}
export function renderGuide(container, lang, method, onChange, onBack) {
  const t = (k) => strings[lang][k],
    ja = lang === 'ja',
    section = guideSection(method, lang);
  const workshop = ja
    ? '最小内角15°、面積25 mm²、頂点間隔3.2 mmを工作上の目安にしています。寸法を大きくすると面積と間隔が増えます。角度を広げるには重ね方を変えます。磁石やボタンが入るかは、書き出す際に個々のパネルで確認します。'
    : 'The workshop guides are 15° for angles, 25 mm² for panel area and 3.2 mm between vertices. A larger square increases area and spacing. Change the overlay to widen an angle. Each magnetic or snap joint also needs room on its individual panel.';
  container.innerHTML =
    '<h1 class="guide-title">' +
    t('guide') +
    '</h1><p class="guide-intro">' +
    t('guideIntro') +
    '</p>' +
    '<div class="guide-tabs" role="group" aria-label="' +
    t('whichGuide') +
    '">' +
    methods
      .map(
        (k) =>
          '<button class="chip" data-guide="' +
          k +
          '" aria-pressed="' +
          (method === k) +
          '">' +
          t(k === 'paper' ? 'guidePaper' : k) +
          '</button>',
      )
      .join('') +
    '</div>' +
    '<article class="panel method-guide"><div class="guide-materials"><h2>' +
    section.title +
    '</h2>' +
    diagram(method, lang) +
    '<h3>' +
    (ja ? '用意するもの' : 'What you need') +
    '</h3><p>' +
    section.materials +
    '</p><h3>' +
    (ja ? '調整のポイント' : 'Checking the fit') +
    '</h3><p>' +
    section.fit +
    '</p></div>' +
    '<div class="guide-steps"><h3>' +
    (ja ? '手順' : 'Assembly steps') +
    '</h3><ol>' +
    section.steps.map((step) => '<li>' + step + '</li>').join('') +
    '</ol></div></article>' +
    '<details class="panel workshop-note"><summary>' +
    t('fitGuide') +
    '</summary><p>' +
    workshop +
    '</p></details>' +
    '<article class="panel guide-research"><div><h2>' +
    t('sourceTitle') +
    '</h2><p>' +
    t('sourceText') +
    '</p><a href="https://archive.bridgesmathart.org/2022/bridges2022-371.html" target="_blank" rel="noreferrer">' +
    t('paperLink') +
    '</a></div><img src="./web/assets/prototypes.webp" alt="' +
    t('photo') +
    '" loading="lazy"></article>' +
    '<button class="btn primary return-btn" id="guide-back">' +
    t('back') +
    '</button>';
  container
    .querySelectorAll('[data-guide]')
    .forEach((button) => (button.onclick = () => onChange(button.dataset.guide)));
  container.querySelector('#guide-back').onclick = onBack;
}
