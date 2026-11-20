const paths = {
  globe: '<circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/><path d="M3 12h18"/>',
  cube: '<path d="m12 2.5 8.5 4.8v9.4L12 21.5l-8.5-4.8V7.3Z"/><path d="m3.5 7.3 8.5 5 8.5-5M12 12.3v9"/>',
  net: '<path d="M9 2h6v5h5v6h-5v9H9v-9H4V7h5z"/><path d="M9 7h6M9 13h6M9 17h6M9 7v6m6-6v6" stroke-dasharray="1.5 2"/>',
  expand: '<path d="M9 4H4v5m11-5h5v5M4 15v5h5m11-5v5h-5"/>',
  collapse: '<path d="M4 9h5V4m6 0v5h5M4 15h5v5m11-5h-5v5"/>',
  minus: '<path d="M5 12h14"/>',
  plus: '<path d="M5 12h14M12 5v14"/>',
  focus: '<path d="M8 4H4v4m12-4h4v4M4 16v4h4m12-4v4h-4"/><circle cx="12" cy="12" r="3.4"/>',
  play: '<path fill="currentColor" stroke="none" d="M7 4.8c0-.8.8-1.2 1.5-.8l11 7.2a1 1 0 0 1 0 1.6L8.5 20c-.7.4-1.5 0-1.5-.8Z"/>',
  pause: '<path fill="currentColor" stroke="none" d="M6 4h4v16H6zm8 0h4v16h-4z"/>',
  tune: '<path d="M4 7h7m4 0h5M4 17h3m4 0h9"/><circle cx="13" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
  collection:
    '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
  left: '<path d="m14 6-6 6 6 6"/>',
  right: '<path d="m10 6 6 6-6 6"/>',
  download: '<path d="M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4"/>',
  link: '<path d="m10 14 4-4m-1.3 6.5-2.1 2.1a4.6 4.6 0 0 1-6.5-6.5l3.3-3.3a4.6 4.6 0 0 1 6.3-.1m-2.4-1.2 2.1-2.1a4.6 4.6 0 0 1 6.5 6.5l-3.3 3.3a4.6 4.6 0 0 1-6.3.1"/>',
  external: '<path d="M13 4h7v7m0-7L9 15M9 6H4v14h14v-5"/>',
  close: '<path d="m6 6 12 12m0-12L6 18"/>',
  check: '<path d="m4.5 12 5 5 10-11"/>',
  paper: '<path d="M5 3h9l5 5v13H5zM14 3v6h5M8 13h8m-8 4h6"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5zm-9 9 9 5 9-5m-18 4 9 5 9-5"/>',
  hand: '<path d="M8 11V6a1.5 1.5 0 0 1 3 0v4-6a1.5 1.5 0 0 1 3 0v6-4a1.5 1.5 0 0 1 3 0v6-3a1.5 1.5 0 0 1 3 0v6a7 7 0 0 1-7 7h-1c-2 0-4-1.4-5.4-3.2l-4-5.2a1.7 1.7 0 0 1 2.5-2.2l2.9 2.4z"/>',
};
export function icon(name) {
  return (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    paths[name] +
    '</svg>'
  );
}
export function applyIcons(root = document) {
  root.querySelectorAll('[data-icon]').forEach((el) => (el.outerHTML = icon(el.dataset.icon)));
}
