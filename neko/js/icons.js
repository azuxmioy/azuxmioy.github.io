const paths = {
  music: '<path d="M9 18V5l11-2v13M9 8l11-2"/><ellipse cx="6" cy="18" rx="3" ry="2.5"/><ellipse cx="17" cy="16" rx="3" ry="2.5"/>',
  wind: '<path d="M3 8h12c6 0 5-7 1-5M2 12h18M4 16h10c6 0 5 7 1 5"/>',
  orbit: '<ellipse cx="12" cy="12" rx="10" ry="4.5" transform="rotate(-28 12 12)"/><path d="m18 3 3 2-3 2"/><circle cx="12" cy="12" r="2"/>',
  cat: '<path d="M4 17V4l6 4h4l6-4v13c-4 4-12 4-16 0Z"/><path d="M8 13h1m6 0h1m-6 4q2 2 4 0"/>',
  paw: '<ellipse cx="12" cy="16" rx="5" ry="4"/><ellipse cx="5" cy="10" rx="2" ry="2.5"/><ellipse cx="10" cy="6" rx="2" ry="2.5"/><ellipse cx="16" cy="7" rx="2" ry="2.5"/><ellipse cx="20" cy="12" rx="2" ry="2.5"/>',
  arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>',
  smoke: '<path d="M3 15h16v4H3zm13 0v4m2-8c-4-4 3-4 0-8m-5 8c-4-4 3-4 0-8"/>',
  heart: '<path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1.1-1.1a5.5 5.5 0 0 0-7.8 7.8L12 21l8.8-8.6a5.5 5.5 0 0 0 0-7.8Z"/>',
  broom: '<path d="m15 3-5 11m-4-1 8 4-3 5-9-4 4-5Zm1 2-2 4m5-3-2 4"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>',
  moon: '<path d="M20.5 14A9 9 0 0 1 10 3.5 9 9 0 1 0 20.5 14Z"/>',
  volume: '<path d="M4 9h4l5-4v14l-5-4H4Zm12-1a6 6 0 0 1 0 8m3-11a10 10 0 0 1 0 14"/>',
  muted: '<path d="M4 9h4l5-4v14l-5-4H4Zm13 0 5 6m0-6-5 6"/>',
  reset: '<path d="M4 10a8 8 0 1 1 1 8M4 4v6h6"/>',
  expand: '<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/>',
  chevron: '<path d="m9 5 7 7-7 7"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  close: '<path d="m6 6 12 12M6 18 18 6"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.2"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  door: '<path d="M4 21h16M7 21V3h10v18m-4-10v2"/>',
  sparkles: '<path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Zm7 0v4m-2-2h4"/>',
};

export function icon(name, className = '') {
  return `<svg class="icon ${className}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.cat}</svg>`;
}
