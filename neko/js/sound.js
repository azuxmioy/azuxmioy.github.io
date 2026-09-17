const MAX_VOICES = 36;

/** Original, gesture-unlocked cartoon sound effects synthesized with Web Audio. */
export class CartoonAudio {
  constructor() {
    this.context = null;
    this.master = null;
    this.noiseBuffer = null;
    this.voices = new Set();
    this._muted = false;
    this._disposed = false;
  }

  get muted() {
    return this._muted;
  }

  /** Call and await this from the first real click; construction never plays audio. */
  async unlock() {
    if (this._disposed) return false;
    try {
      if (!this.context) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) return false;
        this.context = new AudioContextClass();
        this.master = this.context.createGain();
        this.master.gain.value = this._muted ? 0 : 0.62;

        // Soften peaks when several friends make noises at the same time.
        this.compressor = this.context.createDynamicsCompressor();
        this.compressor.threshold.value = -15;
        this.compressor.knee.value = 20;
        this.compressor.ratio.value = 5;
        this.compressor.attack.value = 0.004;
        this.compressor.release.value = 0.16;
        this.master.connect(this.compressor).connect(this.context.destination);

        this.noiseBuffer = this.context.createBuffer(1, this.context.sampleRate * 2, this.context.sampleRate);
        const samples = this.noiseBuffer.getChannelData(0);
        for (let index = 0; index < samples.length; index += 1) samples[index] = Math.random() * 2 - 1;
      }
      if (this.context.state !== 'running') await this.context.resume();
      return !this._disposed && this.context.state === 'running';
    } catch {
      return false;
    }
  }

  setMuted(muted) {
    if (this._disposed) return;
    this._muted = Boolean(muted);
    if (!this.context || !this.master) return;
    const now = this.context.currentTime;
    this.master.gain.cancelScheduledValues(now);
    this.master.gain.setTargetAtTime(this._muted ? 0 : 0.62, now, 0.012);
    if (this._muted) {
      // Discard queued punchlines as well, so unmuting does not replay old cues.
      for (const voice of this.voices) this._removeVoice(voice);
    }
  }

  play(kind) {
    if (this._disposed || this._muted || this.context?.state !== 'running') return;

    switch (kind) {
      case 'lighter':
        this._tone([1600, 650], 0.055, 0.13, 'triangle');
        this._noise(0.045, 0.24, 3800, 'highpass', 0.012);
        this._tone([2800, 1200], 0.04, 0.075, 'sine', 0.065);
        this._noise(0.22, 0.18, 1700, 'bandpass', 0.095, { attack: 0.025, q: 0.65 });
        break;
      case 'puff':
        this._noise(0.64, 0.34, 1250, 'lowpass', 0, { attack: 0.17, endFrequency: 380 });
        this._tone([180, 115], 0.18, 0.045, 'sine', 0.08);
        break;
      case 'cough':
        [0, 0.205, 0.455].forEach((delay, index) => {
          this._noise(index === 2 ? 0.235 : 0.145, 0.55, 680 - index * 95, 'bandpass', delay, { attack: 0.012, q: 0.8 });
          this._tone([145 - index * 12, 82], 0.14, 0.09, 'sawtooth', delay, { cutoff: 490 });
        });
        break;
      case 'boing':
        this._tone([150, 770, 245, 580, 190, 370, 110], 0.68, 0.24, 'triangle', 0, { attack: 0.014, hold: 0.3 });
        this._tone([230, 880, 290, 690, 220, 460, 140], 0.62, 0.075, 'sine', 0.012, { hold: 0.22 });
        break;
      case 'bonk':
        this._tone([340, 90], 0.125, 0.31, 'sine');
        this._tone([720, 480], 0.075, 0.11, 'triangle');
        this._noise(0.032, 0.17, 1650, 'lowpass');
        break;
      case 'gasp':
        this._tone([360, 560, 1200, 1470], 0.34, 0.16, 'sine', 0, { attack: 0.035, hold: 0.21 });
        this._noise(0.28, 0.26, 1800, 'bandpass', 0.015, { attack: 0.13, q: 0.65, endFrequency: 2800 });
        break;
      case 'can':
        this._noise(0.05, 0.33, 3200, 'highpass');
        this._tone([960, 680], 0.16, 0.18, 'triangle', 0.015);
        this._tone([1560, 1510], 0.23, 0.06, 'sine', 0.018);
        this._tone([420, 320], 0.25, 0.14, 'sine', 0.024);
        this._noise(0.33, 0.095, 2400, 'highpass', 0.08, { attack: 0.05 });
        break;
      case 'hiccup':
        this._tone([190, 550, 670, 210], 0.165, 0.24, 'triangle', 0, { attack: 0.008, hold: 0.08 });
        this._tone([480, 1150, 580], 0.09, 0.065, 'sine', 0.025);
        break;
      case 'meow':
        this._meow();
        break;
      case 'fan':
        this._noise(0.92, 0.4, 320, 'lowpass', 0, { attack: 0.22, hold: 0.35, endFrequency: 1500 });
        this._noise(0.58, 0.085, 2400, 'bandpass', 0.12, { attack: 0.16, q: 1.5, endFrequency: 700 });
        break;
      case 'switch':
        this._tone([840, 420], 0.065, 0.14, 'triangle');
        this._noise(0.016, 0.1, 2100, 'highpass');
        break;
      case 'chaos':
        this._tone([233, 220, 175, 164, 130, 116, 103, 98], 1.18, 0.29, 'sawtooth', 0, { attack: 0.035, hold: 0.7, cutoff: 720 });
        this._tone([116, 110, 87, 82, 65, 58, 51, 49], 1.18, 0.16, 'triangle', 0, { attack: 0.04, hold: 0.6 });
        this._tone([104, 112, 101, 109, 98], 0.3, 0.12, 'sawtooth', 0.87, { attack: 0.015, cutoff: 480 });
        break;
      default:
        break;
    }
  }

  _envelope(duration, volume, start, { attack = 0.007, hold = 0 } = {}) {
    const envelope = this.context.createGain();
    const rise = Math.min(attack, duration * 0.45);
    const sustain = Math.max(rise, Math.min(hold, duration * 0.8));
    envelope.gain.setValueAtTime(0, start);
    envelope.gain.linearRampToValueAtTime(volume, start + rise);
    if (sustain > rise) envelope.gain.setValueAtTime(volume, start + sustain);
    envelope.gain.exponentialRampToValueAtTime(0.0001, start + duration);
    envelope.gain.linearRampToValueAtTime(0, start + duration + 0.02);
    envelope.connect(this.master);
    return envelope;
  }

  _tone(pitches, duration, volume, type = 'sine', delay = 0, options = {}) {
    const context = this.context;
    const start = context.currentTime + 0.006 + delay;
    const oscillator = context.createOscillator();
    oscillator.type = type;
    oscillator.frequency.setValueAtTime(pitches[0], start);
    for (let index = 1; index < pitches.length; index += 1) {
      oscillator.frequency.exponentialRampToValueAtTime(pitches[index], start + duration * index / (pitches.length - 1));
    }
    const envelope = this._envelope(duration, volume, start, options);
    const nodes = [oscillator, envelope];
    if (options.cutoff) {
      const filter = context.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.value = options.cutoff;
      filter.Q.value = 0.6;
      oscillator.connect(filter).connect(envelope);
      nodes.push(filter);
    } else {
      oscillator.connect(envelope);
    }
    this._track(nodes, [oscillator], start + duration + 0.025);
    oscillator.start(start);
    oscillator.stop(start + duration + 0.025);
  }

  _noise(duration, volume, frequency, type, delay = 0, options = {}) {
    const context = this.context;
    const start = context.currentTime + 0.006 + delay;
    const source = context.createBufferSource();
    source.buffer = this.noiseBuffer;
    source.loop = true;
    const filter = context.createBiquadFilter();
    filter.type = type;
    filter.Q.value = options.q ?? 0.7;
    filter.frequency.setValueAtTime(frequency, start);
    if (options.endFrequency) filter.frequency.exponentialRampToValueAtTime(options.endFrequency, start + duration);
    const envelope = this._envelope(duration, volume, start, options);
    source.connect(filter).connect(envelope);
    this._track([source, filter, envelope], [source], start + duration + 0.025);
    source.start(start, Math.random());
    source.stop(start + duration + 0.025);
  }

  _meow() {
    const context = this.context;
    const start = context.currentTime + 0.006;
    const duration = 0.61;
    const voice = context.createOscillator();
    voice.type = 'sawtooth';
    voice.frequency.setValueAtTime(370, start);
    voice.frequency.exponentialRampToValueAtTime(610, start + 0.14);
    voice.frequency.exponentialRampToValueAtTime(450, start + 0.29);
    voice.frequency.exponentialRampToValueAtTime(230, start + duration);
    const envelope = this._envelope(duration, 0.33, start, { attack: 0.055, hold: 0.2 });
    const nodes = [voice, envelope];

    [[950, 510, 0.85], [1850, 990, 0.32]].forEach(([from, to, level]) => {
      const formant = context.createBiquadFilter();
      formant.type = 'bandpass';
      formant.Q.value = 2.1;
      formant.frequency.setValueAtTime(from, start);
      formant.frequency.exponentialRampToValueAtTime(to, start + duration);
      const gain = context.createGain();
      gain.gain.value = level;
      voice.connect(formant).connect(gain).connect(envelope);
      nodes.push(formant, gain);
    });

    const vibrato = context.createOscillator();
    vibrato.frequency.value = 7;
    const vibratoAmount = context.createGain();
    vibratoAmount.gain.value = 9;
    vibrato.connect(vibratoAmount).connect(voice.frequency);
    nodes.push(vibrato, vibratoAmount);
    this._track(nodes, [voice, vibrato], start + duration + 0.025);
    voice.start(start);
    vibrato.start(start);
    voice.stop(start + duration + 0.025);
    vibrato.stop(start + duration + 0.025);
  }

  _track(nodes, sources, end) {
    while (this.voices.size >= MAX_VOICES) this._removeVoice(this.voices.values().next().value);
    const voice = { nodes, sources, timer: null, remaining: sources.length };
    this.voices.add(voice);
    for (const source of sources) {
      source.onended = () => {
        voice.remaining -= 1;
        if (voice.remaining === 0) this._removeVoice(voice);
      };
    }
    // Fallback cleanup also covers suspended tabs and interrupted audio contexts.
    voice.timer = window.setTimeout(() => this._removeVoice(voice), Math.max(0, end - this.context.currentTime) * 1000 + 180);
  }

  _removeVoice(voice) {
    if (!voice || !this.voices.delete(voice)) return;
    window.clearTimeout(voice.timer);
    for (const source of voice.sources) {
      source.onended = null;
      try { source.stop(); } catch { /* The source may have already ended. */ }
    }
    for (const node of voice.nodes) node.disconnect();
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    for (const voice of this.voices) this._removeVoice(voice);
    this.master?.disconnect();
    this.compressor?.disconnect();
    this.noiseBuffer = null;
    if (this.context && this.context.state !== 'closed') this.context.close().catch(() => {});
  }
}
