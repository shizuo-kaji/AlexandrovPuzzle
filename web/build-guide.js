// Instructions use millimeters throughout. The 20 mm rule in each PDF also
// checks the alignment template used by the three flat-printed kits.
const en = {
  paper: {
    title: 'Fold a paper model',
    materials: 'Light cardstock, a ruler, a blunt scoring tool, scissors and removable tape.',
    steps: [
      'Choose a square size and a form to assemble. For a first fold, try C0 at 40 mm. Download the PDF and print on the chosen A4 or Letter paper at 100%. Measure the 20 mm rule before cutting.',
      'For a combination, cut one cross from the combined sheet and keep the individual pattern pages as guides. Cut the solid outline. Lightly score the dashed creases with the ruler and blunt tool.',
      'Follow the simulation for one complete target. Bring the panels around the central part with the printed face outside. Keep the folds belonging only to other targets flat.',
      'Use the letters on this target’s page. Bring each pair of edges marked A–G together, endpoint to endpoint. Tape each seam as it becomes accessible. To change forms, peel off the perimeter tape, open the cross and use the next target’s page.',
    ],
    fit: 'If the rule measures anything other than 20 mm, choose actual size and print again. A partial crease selection makes a practice sheet. To close a particular form, include all its required creases.',
  },
  tape: {
    title: 'Join printed panels with tape',
    materials: 'PLA, a 0.4 mm nozzle, thin flexible tape and the actual-size paper template.',
    steps: [
      'Print the PDF at 100% and check its 20 mm rule. The flat cross spans three square widths by four. Check that this footprint fits your bed. For separate print batches, keep track of each part on the template.',
      'Slice the panel STL in millimeters, flat in the supplied layout. Start with 0.2 mm layers and the PLA profile for your printer. Panels are 1.2 mm thick with chamfered edges facing up. Try two adjacent plates and a short hinge first.',
      'Flip the layout as a group so the smooth bed faces are up. Match the long arm and the asymmetric diagonals to the printed template. Center the plates in their outlines, keeping the designed gaps, nominally 3.2 mm.',
      'Bridge each internal gap with a strip of tape along its length on the smooth side. Stop each strip short of the corners so neighboring hinges can move. Flex the test hinge through the angle needed for your model.',
      'Follow one target’s simulation with the tape outside. Tape the seven matching A–G perimeter pairs from that target’s guide. To change forms, release the perimeter tape, lay the hinged sheet flat and fold the next crease set.',
    ],
    fit: 'If a hinge binds, restore the template spacing and reapply the strip without tension. The app flags angles under 15°, areas under 25 mm² and vertex spacing under 3.2 mm. The kit also checks the bevels and remaining panel width. Increase the square size or remove some creases if the plates are too small.',
  },
  magnet: {
    title: 'Make detachable magnetic bridges',
    materials:
      'PLA panels, axially magnetized Ø4 × 1 mm discs, two-part epoxy compatible with the panels and magnet coating, thin woven fabric, needle and thread. Each bridge uses four discs, two in the plates and two in the fabric.',
    steps: [
      'Print the two-plate fit sample first. Use a 0.4 mm nozzle, 0.2 mm layers and the PLA profile, with pockets facing up. The plates are 2.0 mm thick. Each blind pocket is Ø4.3 mm and 1.1 mm deep.',
      'Dry-fit a disc. Mark one pole on a stack and separate the discs one at a time. Epoxy one disc in each panel pocket with the same pole exposed throughout. Seat the discs flush or slightly below the top and allow the epoxy to cure.',
      'Make a bridge from two layers of thin fabric, about 6 × 17 mm, with rounded ends. Stitch a closed pocket for one disc at each end, 11 mm center to center. Both ends must attract the plate magnets when the same face of the strip is down. Leave the fabric between the pockets flexible.',
      'Test the bridge across the sample’s 4.4 mm gap. Fold the plates, reopen them and peel the bridge off by one end. Once the fit works, print the full panel STL flat. Print magnet-placement.svg at 100%, lay the socketed faces up on the map and join the two sockets with the same H number at each internal fold.',
      'Fold with the bridges outside. Find your target in the table on magnet-placement.svg. For each seam A–G, connect the two E-numbered perimeter sockets in that row. Perimeter fittings are planned for targets with every crease selected. Peel those bridges off and open the sheet to choose another target.',
      'Keep loose discs and magnetic parts out of children’s reach. Inspect the stitching and adhesive before each use. If someone may have swallowed a magnet, seek immediate medical attention.',
    ],
    fit: 'If a disc sits proud, clear the pocket and check the printed diameter and depth on the sample. If a bridge lifts, verify its polarity, thin the fabric over the discs and retest. A socket-placement warning means the selected geometry needs more room. Increase the square size or simplify the crease set.',
  },
  snap: {
    title: 'Fit removable TPU snap links',
    materials:
      'PLA or PETG for the panels and TPU 95A for the links. Print the rigid and flexible STL files separately.',
    steps: [
      'Print the two rigid sample plates and one TPU link first. Use a 0.4 mm nozzle, 0.2 mm layers and the material profiles for your printer. Print flat as supplied, with studs up and supports off.',
      'Check the sample dimensions. The plates are 1.2 mm thick. Each stud rises 2.0 mm above its panel, with a Ø3.0 mm stem and Ø3.9 mm head. The link is 0.8 mm thick with two Ø3.3 mm holes on 9 mm centers.',
      'Press each eye over a stud until it seats beneath the head. Flex the sample toward a full fold, reopen it and lift one eye off while supporting the stud. If the fit is tight or loose, adjust the slicer’s hole compensation on the TPU link and retest at the original center spacing.',
      'Print the full panel and link files in millimeters. Print snap-placement.svg at 100%. Arrange the panels stud-side up over their outlines, leaving the 2.8 mm gaps. Fit a link across each green line, including both pairs where a long fold has two links.',
      'Follow the simulation for a complete target with the links outside. For each seam A–G, read the two blue stud numbers in your target’s row on snap-placement.svg and link that pair. Perimeter fittings are planned for targets with every crease selected. To change shape, support each stud as you peel off the perimeter links, unfold the sheet and use the next target’s guide.',
    ],
    fit: 'The layout checks room for the studs and link eyes. If they do not fit, increase the square size or remove some creases and regenerate the files. Verify retention and release on the sample with your own filament before printing the full set.',
  },
  solid: {
    title: 'Print the finished form',
    materials:
      'An FDM printer, PLA or PETG and a slicer. The STL is one fixed solid with recessed lines along the panel boundaries.',
    steps: [
      'Select a complete assembly target and set the square size in millimeters. Play the fold to 100% and rotate the model to check the form. Additional selected crease patterns appear as grooves on its surface.',
      'Import the finished STL at 100% in millimeters. It opens with a broad panel on the bed. Start with a 0.4 mm nozzle, 0.2 mm layers and your usual material profile. Inspect the first layer, the grooves and the overhangs in the slicer preview.',
      'Use a brim if the contact area needs more adhesion and add supports where the preview indicates unsupported overhangs. Print, let the bed cool and remove the model. The Q targets are two-sided, 2.4 mm thick reliefs of the flat folded form.',
    ],
    fit: 'The groove shoulders extend about 0.65 mm into each panel and narrow near small tips. If the slicer loses fine lines, increase the square size in the studio or reduce the overlays, then export again. The separate assembly kits provide movable hinges.',
  },
};

const ja = {
  paper: {
    title: '紙で折る',
    materials: '薄手の工作用紙、定規、先の丸いヘラ、はさみ、はがせるテープ。',
    steps: [
      '正方形の一辺と組み立てる形を選びます。初めてなら一辺40 mmの立方体 C0 から試せます。PDFを指定のA4またはレター用紙に原寸（100%）で印刷し、目盛りの両端が20 mmあることを定規で確かめます。',
      '複数の図を重ねた場合は、重ね図から十字形を1枚切り出し、各パターンのページを手元に置きます。外周の実線に沿って切り、破線には定規とヘラで軽く折り筋をつけます。',
      '組み立てる形のシミュレーションを見ながら、印字面が外側になるよう中央から折り起こします。ほかの形で使う折り筋は平らに保ちます。',
      'その形のページで A〜G の記号を確認し、同じ文字の辺の両端をそろえ、合わせやすい順にテープで留めます。折り替えるときは外周のテープをはがして十字形に戻し、次の形の図を使います。',
    ],
    fit: '目盛りが20 mmにならないときは「実際のサイズ」を選んで刷り直してください。折り線を一部だけ選んだ図は折り方の試作に使えます。立体を閉じるには、その形に必要な折り線をすべて含めます。',
  },
  tape: {
    title: 'テープでパネルをつなぐ',
    materials: 'PLA、0.4 mmノズル、薄くしなやかなテープ、原寸の型紙。',
    steps: [
      '型紙を100%で印刷し、20 mmの目盛りを確認します。十字形の外寸は正方形の一辺の横3倍・縦4倍です。造形台に収まるか確かめ、分けて印刷する場合は各部品の位置を型紙に控えます。',
      'パネルSTLをmm単位で読み込み、配置を保って平置きします。まずPLA用の設定と積層0.2 mmで、隣り合う2枚を試し刷りします。厚さは1.2 mmで、角を落とした面が上になります。',
      '配置全体を裏返して造形台に接していた平滑面を上にします。長い腕と左右非対称の斜線を型紙に合わせ、各パネルを輪郭の中央に置きます。パネル間のすき間は標準3.2 mmです。',
      '平滑面に、各すき間をまたいで折り線に沿うテープを貼ります。端を頂点の少し手前で止めると、隣のヒンジも動かせます。試した2枚を必要な角度まで折ってから、全体をつなぎます。',
      'テープ面を外側にして、目標の形のシミュレーションに沿って折ります。その形の図を見て A〜G の同じ文字の辺をテープで留めます。折り替えるときは外周だけをはがし、平らに戻します。',
    ],
    fit: '折りが固いときは型紙どおりのすき間を取り、テープを引っ張らずに貼り直します。画面の注意の目安は内角15°、面積25 mm²、頂点間隔3.2 mm未満です。キットの出力時には面取り後の幅も調べます。部品が小さすぎる場合は一辺を長くするか、折り線を減らします。',
  },
  magnet: {
    title: '布と磁石で着脱式のヒンジを作る',
    materials:
      'PLAパネル、厚み方向に着磁した直径4 mm・厚さ1 mmの円板磁石、パネル材と磁石の表面に使える二液エポキシ接着剤、薄い織布、針と糸。布のブリッジ1本につき、布側とパネル側に各2個の磁石を使います。',
    steps: [
      'まず2枚組の試験片を、くぼみを上にして印刷します。PLA用の設定で、ノズル径0.4 mm・積層0.2 mmが出発点です。パネルは厚さ2.0 mm、くぼみは直径4.3 mm・深さ1.1 mmです。',
      '磁石が入るか仮合わせします。重ねた磁石の片方の極に印をつけ、1個ずつ横にずらして外します。全パネルで同じ極が表に出るように1個ずつ接着し、パネル面から突き出さない位置で硬化させます。',
      '幅約6 mm・長さ約17 mmの薄い布を2枚重ね、両端を丸くします。磁石の中心間を11 mmにとり、両端に1個ずつ入れて、端ごとに袋を縫い閉じます。布の同じ面をパネルに向けたとき、両端とも吸着する向きです。中央は曲がるように残します。',
      '試験片の4.4 mmのすき間を布でつなぎ、折る・開く・布を片側からはがす動作を試します。保持できたら全体を平置きで印刷します。magnet-placement.svg を100%で印刷し、くぼみを上にして配置図に並べます。内側の折り線では、同じH番号の2か所を布でつなぎます。',
      '布が外側になるように折り、配置図の表で作る形の行を見ます。A〜G の接合ごとに、指定された2つのE番号のくぼみを布でつなぎます。外周の接合位置は、折り線をすべて選んだ形について用意されます。外周の布を片側からはがし、十字形に戻すと別の形に折り替えられます。',
      '小さな磁石と磁石入りの部品は子どもの手が届かない所に保管し、使う前に縫い目と接着を確かめます。磁石を飲み込んだ疑いがあるときはすぐに医療機関を受診してください。',
    ],
    fit: '磁石が浮くときはくぼみの異物を除き、試験片の内径と深さを測ります。吸着が弱いときは両端の極性を確かめ、磁石を覆う布を薄くして再度試します。くぼみを配置できない場合は、一辺を長くするか折り線を減らします。',
  },
  snap: {
    title: 'TPUリンクをボタンにはめる',
    materials:
      'パネル用のPLAまたはPETGと、リンク用の硬度95AのTPU。硬い部品と柔らかい部品は別々のSTLで印刷します。',
    steps: [
      'はじめに硬い試験片2枚とTPUリンク1本を印刷します。ノズル径0.4 mm・積層0.2 mmを目安に材料別の設定を使い、支え材をオフにして平置きします。パネルのボタンは上向きです。',
      '試験片の寸法を確かめます。パネル厚は1.2 mm。ボタンは根元の軸径3.0 mm、頭の最大径3.9 mm、パネル面からの高さ2.0 mmです。リンクは厚さ0.8 mmで、直径3.3 mmの穴が中心間隔9 mmで開いています。',
      'リンクの穴をひとつずつボタンの頭の下まではめます。試験片を深く折ってから開き、ボタンを支えて片方の穴をめくり外します。固すぎるときはTPU側の穴を少し広げ、緩いときは少し狭めるよう穴径を補正し、中心間隔を保って試し刷りします。',
      '全体のパネルとリンクをmm単位で印刷します。snap-placement.svg を100%で印刷し、ボタンを上にして配置図に合わせます。パネル間に2.8 mmのすき間を残し、緑の線1本につきリンクを1本はめます。長い折り線に2組のボタンがある場合は両方をつなぎます。',
      'リンクが外側になるようシミュレーションに沿って折ります。配置図の表で作る形の行を見ます。A〜G の接合ごとに、指定された2つの青い番号のボタンをリンクでつなぎます。外周の接合位置は、折り線をすべて選んだ形について用意されます。折り替えるときは外周のボタンを支えながらリンクを片端ずつ外し、平らに戻します。',
    ],
    fit: '出力時にはボタンとリンクの穴の周りに必要な余白を調べます。収まらない場合は一辺を長くするか折り線を減らし、再出力してください。実際の保持力と着脱の固さは、全体を刷る前に使用するフィラメントで試します。',
  },
  solid: {
    title: '完成形を一体で印刷する',
    materials:
      'FDMプリンター、PLAまたはPETG、スライサー。完成形STLにはパネルの境界が溝として入っています。',
    steps: [
      '目標の形の折り線をすべて選び、正方形の一辺をmm単位で指定します。組み立てを100%まで進め、回転して仕上がりの形を確認します。重ねたほかのパターンの折り線も表面に刻まれます。',
      '完成形STLをmm単位・100%で読み込みます。広い面が造形台に接する向きで開きます。ノズル径0.4 mm・積層0.2 mmを目安に材料別の設定を使い、スライス後に最初の層、溝、張り出しを確認します。',
      '接地面が小さい場合はブリムを付け、張り出しに支えが必要ならサポートを追加します。印刷後は造形台が冷めてから外します。平らな二重被覆四角形のQ系は、表裏に折り線のある厚さ2.4 mmの板として出力されます。',
    ],
    fit: '溝はパネル境界から片側約0.65 mmの幅を使い、小さい角では幅が狭くなります。スライス後に細い線が消える場合は、一辺を長くするか重ねる線を減らして再出力します。折って遊ぶ模型には、ヒンジを備えた組み立てキットを使います。',
  },
};

export function guideSection(method, lang = 'en') {
  const guide = (lang === 'ja' ? ja : en)[method];
  if (!guide) throw new RangeError('Unknown build method ' + method);
  return guide;
}

export function instructionsText(method, lang, details = {}) {
  const jp = lang === 'ja',
    guide = guideSection(method, lang);
  const lines = ['Alexandrov Puzzle — ' + guide.title, ''];
  if (details.coupon) lines.push(jp ? '実寸の接合試験片' : 'Actual-size joint sample');
  const id = (p) => (typeof p === 'string' ? p : p?.id);
  if (!details.coupon) {
    const patterns = details.patterns?.length ? details.patterns : details.pattern;
    if (patterns) {
      const names = Array.isArray(patterns) ? patterns.map(id).join(' + ') : id(patterns);
      lines.push((jp ? '展開図　' : 'Net · ') + names);
    }
    if (Array.isArray(details.completeTargets)) {
      const complete = details.completeTargets.map(id).join(', ') || (jp ? 'なし' : 'None');
      lines.push((jp ? '全折り線を選んだ形　' : 'Targets with all creases · ') + complete);
    }
    if (details.edgeMm != null)
      lines.push((jp ? '正方形の一辺　' : 'Square edge · ') + details.edgeMm + ' mm');
    if (details.paper) lines.push((jp ? '用紙　' : 'Paper · ') + details.paper);
  }
  if (details.files?.length)
    lines.push('', jp ? '同梱の製作ファイル' : 'Fabrication files', ...details.files);
  const counts = [
    ['panelCount', 'パネル', 'Panels'],
    ['pocketCount', 'パネルの磁石用くぼみ', 'Panel pockets'],
    ['bridgeCount', '布のブリッジ', 'Fabric bridges'],
    ['magnetCount', '円板磁石の総数', 'Total discs'],
    ['hingeLinks', '折り線用TPUリンク', 'Internal TPU links'],
    ['seamLinks', '外周用TPUリンク', 'Perimeter TPU links'],
  ];
  for (const [key, labelJa, labelEn] of counts) {
    if (Number.isFinite(details[key])) lines.push((jp ? labelJa + '　' : labelEn + ' · ') + details[key]);
  }
  lines.push('', jp ? '用意するもの' : 'Materials', guide.materials, '', jp ? '作り方' : 'Assembly');
  guide.steps.forEach((step, i) => lines.push(i + 1 + '. ' + step, ''));
  lines.push(jp ? '寸法と試作' : 'Fit and calibration', guide.fit);
  return lines.join('\n') + '\n';
}
