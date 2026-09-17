const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const random = (min, max) => min + Math.random() * (max - min);

/** A disposable smoke overlay. Coordinates passed to puff() are percentages. */
export class SmokeEffect {
  constructor(container) {
    if (!container || typeof container.append !== 'function') {
      throw new TypeError('SmokeEffect requires an HTML container.');
    }

    this.container = container;
    this.particles = new Set();
    this.destroyed = false;
    this.motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.layer = document.createElement('div');
    this.layer.className = 'smoke-effect';
    this.layer.setAttribute('aria-hidden', 'true');
    Object.assign(this.layer.style, {
      position: 'absolute',
      inset: '0',
      overflow: 'hidden',
      pointerEvents: 'none',
      zIndex: '5',
      borderRadius: 'inherit',
    });
    container.append(this.layer);
  }

  puff({ x = 50, y = 40, intensity = 1 } = {}) {
    if (this.destroyed) return;

    const originX = Number.isFinite(x) ? clamp(x, 0, 100) : 50;
    const originY = Number.isFinite(y) ? clamp(y, 0, 100) : 40;
    const strength = Number.isFinite(intensity) ? clamp(intensity, 0.25, 3) : 1;
    const reduced = this.motionQuery.matches;
    const count = reduced ? 3 : Math.round(13 * strength);
    const bounds = this.container.getBoundingClientRect();
    const size = clamp(bounds.width / 20, 20, 42);

    for (let index = 0; index < count; index += 1) {
      while (this.particles.size >= 84) this._remove(this.particles.values().next().value);

      const node = document.createElement('span');
      const width = size * random(0.7, 1.25);
      const drift = random(-78, -28);
      const rise = clamp(bounds.height * random(0.2, 0.4), 65, 260);
      const curl = random(-32, 32);
      const duration = reduced ? 1500 : random(2800, 4100);
      const delay = reduced ? 0 : index * 45;
      const opacity = reduced ? 0.3 : random(0.23, 0.4);
      Object.assign(node.style, {
        position: 'absolute',
        display: 'block',
        left: `${originX}%`,
        top: `${originY}%`,
        width: `${width}px`,
        height: `${width * 1.55}px`,
        marginLeft: `${-width / 2}px`,
        marginTop: `${-width * 0.65}px`,
        borderRadius: '47% 53% 58% 42% / 65% 39% 61% 35%',
        background: 'radial-gradient(ellipse at 39% 43%, rgba(105, 107, 103, .6) 0%, rgba(139, 142, 134, .46) 29%, rgba(187, 190, 181, .24) 49%, transparent 72%)',
        filter: `blur(${reduced ? 3 : random(2, 4)}px)`,
        opacity: '0',
        transformOrigin: '50% 85%',
        willChange: reduced ? 'opacity' : 'transform, opacity',
      });

      const particle = { node, animation: null, timer: null };
      this.particles.add(particle);
      this.layer.append(node);

      const frames = reduced
        ? [
            { opacity: 0, transform: 'translateY(-12px) scale(.9)' },
            { opacity, offset: 0.25, transform: 'translateY(-12px) scale(.9)' },
            { opacity: 0, transform: 'translateY(-12px) scale(.9)' },
          ]
        : [
            { opacity: 0, transform: 'translate(0, 0) rotate(-8deg) scale(.25)', offset: 0 },
            { opacity, transform: `translate(${drift * 0.15}px, ${-rise * 0.15}px) rotate(${curl}deg) scale(.6, .9)`, offset: 0.2 },
            { opacity: opacity * 0.7, transform: `translate(${drift * 0.7}px, ${-rise * 0.55}px) rotate(${-curl}deg) scale(1, 1.35)`, offset: 0.56 },
            { opacity: 0, transform: `translate(${drift}px, ${-rise}px) rotate(${curl * 1.5}deg) scale(1.65, 1.8)`, offset: 1 },
          ];

      if (typeof node.animate === 'function') {
        particle.animation = node.animate(frames, {
          duration,
          delay,
          easing: 'cubic-bezier(.2, .45, .45, 1)',
          fill: 'both',
        });
        particle.animation.onfinish = () => this._remove(particle);
      } else {
        node.style.opacity = String(opacity);
        node.style.transform = 'translateY(-16px) scale(.9)';
      }

      // Also clean up animations in background tabs, or browsers without WAAPI.
      particle.timer = window.setTimeout(() => this._remove(particle), duration + delay + 100);
    }
  }

  _remove(particle) {
    if (!particle || !this.particles.delete(particle)) return;
    window.clearTimeout(particle.timer);
    if (particle.animation) {
      particle.animation.onfinish = null;
      particle.animation.cancel();
    }
    particle.node.remove();
  }

  clear() {
    for (const particle of this.particles) this._remove(particle);
  }

  destroy() {
    if (this.destroyed) return;
    this.clear();
    this.layer.remove();
    this.destroyed = true;
  }
}
