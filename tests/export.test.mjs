import assert from 'node:assert/strict';
import test from 'node:test';
import { getPattern } from '../web/geometry.js';
import { exportDesign } from '../web/export-design.js';

const patterns = ['O1', 'Q0'];
const selections = Object.fromEntries(patterns.map((id) => [id, getPattern(id).creases.map((c) => c.id)]));
const request = { format: 'kit', target: 'O1', patterns, selections, edgeMm: 60, lang: 'ja', paper: 'A4' };
const decoder = new TextDecoder();

async function unpack(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer()),
    data = new DataView(bytes.buffer),
    files = {};
  let offset = 0;
  while (data.getUint32(offset, true) === 0x04034b50) {
    const length = data.getUint32(offset + 18, true),
      nameLength = data.getUint16(offset + 26, true);
    const start = offset + 30 + nameLength;
    files[decoder.decode(bytes.subarray(offset + 30, start))] = bytes.subarray(start, start + length);
    offset = start + length;
  }
  assert.equal(data.getUint32(offset, true), 0x02014b50);
  assert.equal(data.getUint32(bytes.length - 22, true), 0x06054b50);
  return files;
}
for (const method of ['tape', 'magnet', 'snap']) {
  test(method + ' ships a complete kit for the initial two-target design', async () => {
    const { blob } = await exportDesign({ ...request, method });
    const files = await unpack(blob);
    assert.match(decoder.decode(files['template.pdf']), /\/Count 3\b/);
    assert.match(decoder.decode(files['BUILD-ja.txt']), /60 mm/);
    assert.match(decoder.decode(files['BUILD-en.txt']), /O1, Q0/);
    for (const [name, bytes] of Object.entries(files).filter(([name]) => name.endsWith('.stl'))) {
      const data = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      assert.equal(data.byteLength, 84 + 50 * data.getUint32(80, true), name);
    }
    if (method !== 'tape') {
      assert.ok(files[method + '-placement.svg']);
      assert.ok(Object.keys(files).some((name) => name.startsWith('sample-')));
    }
  });
}
test('a blank outline and incomplete targets can be exported as paper', async () => {
  const empty = await exportDesign({ ...request, patterns: [], selections: {}, format: 'pdf' });
  assert.match(empty.filename, /blank/);
  assert.match(await empty.blob.text(), /\/Count 1\b/);
  const partial = { ...request, format: 'solid', selections: { ...selections, O1: [] } };
  await assert.rejects(exportDesign(partial), /Incomplete/);
  assert.match(await (await exportDesign({ ...partial, format: 'pdf' })).blob.text(), /\/Count 3\b/);
});
