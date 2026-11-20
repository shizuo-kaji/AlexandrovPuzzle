// Uncompressed ZIP keeps binary STLs and the PDF as distinct, named files.
// Only a handful of files are packed, all comfortably below ZIP32 limits.
const encoder = new TextEncoder();
const table = Uint32Array.from({ length: 256 }, (_, i) => {
  let n = i;
  for (let bit = 0; bit < 8; bit++) n = (n >>> 1) ^ (n & 1 ? 0xedb88320 : 0);
  return n >>> 0;
});
function checksum(bytes) {
  let value = 0xffffffff;
  for (const byte of bytes) value = (value >>> 8) ^ table[(value ^ byte) & 255];
  return (value ^ 0xffffffff) >>> 0;
}
function record(size, signature) {
  const data = new Uint8Array(size),
    view = new DataView(data.buffer);
  view.setUint32(0, signature, true);
  return { data, u16: (i, n) => view.setUint16(i, n, true), u32: (i, n) => view.setUint32(i, n, true) };
}
export async function zipFiles(files) {
  const parts = [],
    central = [];
  let offset = 0,
    centralSize = 0;
  for (const [name, file] of Object.entries(files)) {
    const path = encoder.encode(name);
    const bytes = file instanceof Blob ? new Uint8Array(await file.arrayBuffer()) : encoder.encode(file);
    const crc = checksum(bytes),
      local = record(30, 0x04034b50);
    local.u16(4, 20);
    local.u16(6, 0x800);
    local.u16(12, 33);
    local.u32(14, crc);
    local.u32(18, bytes.length);
    local.u32(22, bytes.length);
    local.u16(26, path.length);
    parts.push(local.data, path, bytes);
    const entry = record(46, 0x02014b50);
    entry.u16(4, 20);
    entry.u16(6, 20);
    entry.u16(8, 0x800);
    entry.u16(14, 33);
    entry.u32(16, crc);
    entry.u32(20, bytes.length);
    entry.u32(24, bytes.length);
    entry.u16(28, path.length);
    entry.u32(42, offset);
    central.push(entry.data, path);
    centralSize += 46 + path.length;
    offset += 30 + path.length + bytes.length;
  }
  const end = record(22, 0x06054b50),
    n = Object.keys(files).length;
  end.u16(8, n);
  end.u16(10, n);
  end.u32(12, centralSize);
  end.u32(16, offset);
  return new Blob([...parts, ...central, end.data], { type: 'application/zip' });
}
