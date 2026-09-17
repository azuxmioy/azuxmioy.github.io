const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const random = (min, max) => min + Math.random() * (max - min);
const point = (value = {}) => ({
  x: Number.isFinite(value.x) ? clamp(value.x, 0, 100) : 50,
  y: Number.isFinite(value.y) ? clamp(value.y, 0, 100) : 50,
});

// Small original drawings; all user-visible effect graphics are SVG, never text.
const drawings = {
  stars: '<path d="M40 7 48 29 72 34 53 49 55 73 36 59 14 68 21 44 7 26 31 27Z" fill="#efd36f" stroke="#715843" stroke-width="3" stroke-linejoin="round"/><path d="m39 23 3 11 11 3" fill="none" stroke="#fff6cc" stroke-width="4" stroke-linecap="round"/>',
  drops: '<path d="M43 7C39 22 17 37 19 53c2 17 25 25 38 12C73 48 53 25 43 7Z" fill="#96c6c8" stroke="#52797b" stroke-width="3"/><path d="M29 45c-5 10 1 15 6 16" fill="none" stroke="#e7f6ed" stroke-width="4" stroke-linecap="round"/>',
  notes: '<path d="M30 55V20l33-8v36M30 29l33-8" fill="none" stroke="#697d7a" stroke-width="6" stroke-linejoin="round"/><ellipse cx="21" cy="58" rx="12" ry="8" transform="rotate(-20 21 58)" fill="#697d7a"/><ellipse cx="54" cy="51" rx="12" ry="8" transform="rotate(-20 54 51)" fill="#697d7a"/>',
  bubbles: '<g fill="#dcebe0" fill-opacity=".24" stroke="#82a6a3" stroke-width="3"><circle cx="35" cy="43" r="23"/><circle cx="62" cy="19" r="9"/><circle cx="66" cy="65" r="5"/></g><path d="M21 39c1-7 5-11 12-12" fill="none" stroke="#fffdf0" stroke-width="4" stroke-linecap="round"/>',
  anger: '<g fill="none" stroke="#c26f5c" stroke-width="6" stroke-linecap="round"><path d="M19 11c1 18 6 24 20 25M68 17c-17 1-24 6-25 19M63 70c-1-17-6-24-19-25M12 63c17-1 24-6 25-19"/></g>',
  slipper: '<path d="M14 61c-5-12 0-35 13-47C39 1 60 6 67 21c7 17-4 43-15 49-15 8-32 4-38-9Z" fill="#557d75" stroke="#354f49" stroke-width="3"/><path d="M22 38c3-14 10-23 20-24 12-2 20 7 20 18-12-4-27-1-40 6Z" fill="#a3c7b6" stroke="#354f49" stroke-width="3"/><path d="M24 46c11-7 24-8 34-5" fill="none" stroke="#c8dfc9" stroke-width="3" stroke-linecap="round"/><path d="M27 61c7 4 15 4 21 0" fill="none" stroke="#789c8b" stroke-width="3" stroke-linecap="round"/>',
  can: '<path d="M22 15h36v49c0 10-36 10-36 0Z" fill="#d4977e" stroke="#685e52" stroke-width="3"/><path d="M23 28h34v27H23Z" fill="#eed1a8"/><ellipse cx="40" cy="15" rx="18" ry="7" fill="#c2c9b8" stroke="#685e52" stroke-width="3"/><ellipse cx="40" cy="15" rx="6" ry="3" fill="none" stroke="#777d6f" stroke-width="2"/><path d="M27 62c9 3 18 3 26 0" fill="none" stroke="#f1c4a5" stroke-width="3"/><path d="M37 32c-10 7 10 10 1 18" fill="none" stroke="#bd795f" stroke-width="3" stroke-linecap="round"/>',
  fish: '<path d="M53 38 73 20v40L53 44" fill="#779f99" stroke="#4e6e68" stroke-width="3" stroke-linejoin="round"/><path d="M9 40C19 17 48 17 59 40 48 63 19 63 9 40Z" fill="#a7c7b5" stroke="#4e6e68" stroke-width="3"/><path d="m31 23 10-12 7 17M33 56l9 11 5-14" fill="#d7b587" stroke="#4e6e68" stroke-width="2.5" stroke-linejoin="round"/><circle cx="23" cy="36" r="3" fill="#344d46"/><path d="M34 30c-4 6-4 13 0 19M43 32l-4 8 4 7" fill="none" stroke="#789f8c" stroke-width="2.5" stroke-linecap="round"/>',
  fan: '<path d="M9 34C11 4 67 1 73 33L42 70Z" fill="#edd8b5" stroke="#756250" stroke-width="3" stroke-linejoin="round"/><path d="M10 33c17-14 43-16 62-1L62 44c-12-9-31-9-43 1Z" fill="#ce9781"/><g fill="none" stroke="#a48e6b" stroke-width="2"><path d="m13 23 29 47L27 11M42 70l1-62M42 70l16-57M42 70l28-48"/></g><path d="M41 63v12" stroke="#756250" stroke-width="6" stroke-linecap="round"/>',
  ring: '<ellipse cx="40" cy="40" rx="26" ry="18" fill="none" stroke="#73776c" stroke-opacity=".42" stroke-width="7"/><ellipse cx="39" cy="38" rx="25" ry="18" fill="none" stroke="#e4e5d7" stroke-opacity=".66" stroke-width="3"/>',
  fishRing: '<path d="M10 40c15-27 35-23 46-3l16-13v33L55 43c-17 24-36 18-45-3Z" fill="none" stroke="#7d8175" stroke-opacity=".5" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/><circle cx="25" cy="35" r="2" fill="#8c9083" fill-opacity=".6"/>',
  wind: '<g fill="none" stroke="#78988c" stroke-opacity=".58" stroke-width="2.5" stroke-linecap="round"><path d="M76 21H26C6 21 8 4 20 8M72 39H13M79 55H31C7 55 14 77 25 69"/></g>',
};

/** Disposable SVG gags. The container must establish a positioning context. */
export class GagEffects {
  constructor(container) {
    if (!container || typeof container.append !== 'function') throw new TypeError('GagEffects requires a container.');
    this.container = container;
    this.records = new Set();
    this.destroyed = false;
    this.motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.layer = document.createElement('div');
    this.layer.className = 'gag-effects';
    this.layer.setAttribute('aria-hidden', 'true');
    Object.assign(this.layer.style, {
      position: 'absolute', inset: '0', overflow: 'hidden',
      pointerEvents: 'none', zIndex: '12', borderRadius: 'inherit',
    });
    container.append(this.layer);
  }

  burst(kind, { x = 50, y = 40, count = 5 } = {}) {
    if (this.destroyed || !['stars', 'drops', 'notes', 'bubbles', 'anger'].includes(kind)) return Promise.resolve();
    const origin = point({ x, y });
    const amount = Number.isFinite(count) ? clamp(Math.round(count), 1, 18) : 5;
    const reduced = this.motionQuery.matches;
    const promises = [];
    for (let index = 0; index < amount; index += 1) {
      const angle = (Math.PI * 2 * index / amount) + random(-0.25, 0.25);
      const distance = random(45, 100);
      const dx = Math.cos(angle) * distance;
      const dy = Math.sin(angle) * distance - 24;
      const rotation = random(-35, 35);
      const frames = reduced ? this._fadeFrames() : [
        { opacity: 0, transform: 'scale(.15)', offset: 0 },
        { opacity: 1, transform: `translate(${dx * 0.45}px, ${dy * 0.45}px) rotate(${rotation * 0.4}deg) scale(1.12)`, offset: 0.22 },
        { opacity: 1, transform: `translate(${dx * 0.8}px, ${dy * 0.85}px) rotate(${rotation}deg) scale(.95)`, offset: 0.65 },
        { opacity: 0, transform: `translate(${dx}px, ${dy + (kind === 'drops' ? 28 : -12)}px) rotate(${rotation * 1.4}deg) scale(.7)` },
      ];
      const bounds = this.container.getBoundingClientRect();
      const location = reduced ? point({ x: origin.x + dx / Math.max(bounds.width, 1) * 45, y: origin.y + dy / Math.max(bounds.height, 1) * 45 }) : origin;
      promises.push(this._animate(kind, location, random(24, 37), frames, reduced ? 500 : random(650, 1050), reduced ? 0 : index * 23));
    }
    return Promise.all(promises);
  }

  projectile(kind, from, to, { duration = 650 } = {}) {
    if (this.destroyed || !['slipper', 'can', 'fish', 'fan'].includes(kind)) return Promise.resolve();
    const start = point(from);
    const finish = point(to);
    const reduced = this.motionQuery.matches;
    const bounds = this.container.getBoundingClientRect();
    const dx = (finish.x - start.x) / 100 * bounds.width;
    const dy = (finish.y - start.y) / 100 * bounds.height;
    const arc = clamp(Math.hypot(dx, dy) * 0.3, 40, 155);
    const spin = (dx < 0 ? -1 : 1) * (kind === 'can' ? 540 : 330);
    const frames = reduced ? this._fadeFrames() : Array.from({ length: 9 }, (_, index) => {
      const t = index / 8;
      return {
        offset: t,
        opacity: index === 0 ? 0 : index === 8 ? 0 : 1,
        transform: `translate(${dx * t}px, ${dy * t - Math.sin(Math.PI * t) * arc}px) rotate(${spin * t - 25}deg) scale(${0.78 + Math.sin(Math.PI * t) * 0.28})`,
      };
    });
    const time = Number.isFinite(duration) ? clamp(duration, 150, 4000) : 650;
    return this._animate(kind, reduced ? finish : start, kind === 'slipper' ? 66 : 58, frames, reduced ? 360 : time);
  }

  smokeRing({ x = 50, y = 40, fish = false } = {}) {
    if (this.destroyed) return Promise.resolve();
    const reduced = this.motionQuery.matches;
    const width = this.container.getBoundingClientRect().width;
    const drift = clamp(width * 0.18, 55, 155);
    const frames = reduced ? this._fadeFrames() : [
      { opacity: 0, transform: 'scale(.35)', offset: 0 },
      { opacity: 0.9, transform: `translate(${drift * 0.1}px, -6px) rotate(-8deg) scale(.75)`, offset: 0.18 },
      { opacity: 0.65, transform: `translate(${drift * 0.65}px, -37px) rotate(9deg) scale(1.1)`, offset: 0.64 },
      { opacity: 0, transform: `translate(${drift}px, -72px) rotate(-7deg) scale(1.55)` },
    ];
    return this._animate(fish ? 'fishRing' : 'ring', point({ x, y }), fish ? 78 : 68, frames, reduced ? 650 : 2000);
  }

  wind() {
    if (this.destroyed) return Promise.resolve();
    const reduced = this.motionQuery.matches;
    const width = this.container.getBoundingClientRect().width;
    return Promise.all([30, 48, 66].map((y, index) => {
      const distance = clamp(width * 0.5, 120, 460);
      const frames = reduced ? this._fadeFrames() : [
        { opacity: 0, transform: 'translate(25px, 0) scale(.8)', offset: 0 },
        { opacity: 0.85, transform: `translate(${-distance * 0.25}px, -3px) scale(1.15)`, offset: 0.25 },
        { opacity: 0.65, transform: `translate(${-distance * 0.7}px, 3px) scale(1.05)`, offset: 0.7 },
        { opacity: 0, transform: `translate(${-distance}px, -2px) scale(.9)` },
      ];
      return this._animate('wind', { x: reduced ? 55 : 82, y }, 125, frames, reduced ? 450 : 950, reduced ? 0 : index * 110);
    }));
  }

  _fadeFrames() {
    return [{ opacity: 0 }, { opacity: 1, offset: 0.2 }, { opacity: 0.8, offset: 0.7 }, { opacity: 0 }];
  }

  _animate(kind, location, size, frames, duration, delay = 0) {
    if (this.destroyed) return Promise.resolve();
    while (this.records.size >= 70) this._remove(this.records.values().next().value);
    const node = document.createElement('div');
    node.className = `gag-effect gag-effect--${kind}`;
    node.innerHTML = `<svg viewBox="0 0 80 80" width="100%" height="100%" aria-hidden="true" focusable="false" style="display:block;overflow:visible">${drawings[kind]}</svg>`;
    Object.assign(node.style, {
      position: 'absolute', left: `${location.x}%`, top: `${location.y}%`,
      width: `${size}px`, height: `${size}px`, pointerEvents: 'none',
      transform: 'translate(-50%, -50%)', opacity: '0',
      filter: 'drop-shadow(0 2px 1px rgba(67, 58, 42, .08))',
      willChange: this.motionQuery.matches ? 'opacity' : 'transform, opacity',
    });
    this.layer.append(node);

    return new Promise((resolve) => {
      const record = { node, animation: null, timer: null, resolve };
      this.records.add(record);
      const animationFrames = frames.map((frame) => ({ ...frame, transform: `translate(-50%, -50%) ${frame.transform || ''}` }));
      if (typeof node.animate === 'function') {
        record.animation = node.animate(animationFrames, { duration, delay, easing: 'linear', fill: 'both' });
        record.animation.onfinish = () => this._remove(record);
        record.animation.oncancel = () => this._remove(record);
      } else {
        node.style.opacity = '1';
      }
      record.timer = window.setTimeout(() => this._remove(record), duration + delay + 100);
    });
  }

  _remove(record) {
    if (!record || !this.records.delete(record)) return;
    window.clearTimeout(record.timer);
    if (record.animation) {
      record.animation.onfinish = null;
      record.animation.oncancel = null;
      record.animation.cancel();
    }
    record.node.remove();
    record.resolve();
  }

  clear() {
    for (const record of this.records) this._remove(record);
  }

  destroy() {
    if (this.destroyed) return;
    this.clear();
    this.layer.remove();
    this.destroyed = true;
  }
}
