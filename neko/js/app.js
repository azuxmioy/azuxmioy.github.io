import './components.js';
import { CHARACTER_ACTIONS, PROP_LAYOUT } from './data.js';
import { CHARACTER_INTERACTIONS } from './interactions.js';
import { placeCast } from './layout.js';
import { roomIllustration } from './scene.js';
import { SmokeEffect } from './effects.js';
import { CartoonAudio } from './sound.js';
import { GagEffects } from './gags.js';
import { icon } from './icons.js';

class NekoApp extends HTMLElement {
  connectedCallback() {
    this.guestId = null;
    this.audio = new CartoonAudio();
    this.timers = new Set();
    this.lastActions = new Map();
    this.actors = new Map();
    this.grounds = new Map();
    this.puffNumber = 0;
    this.revision = 0;
    this.busyUntil = 0;
    this.styleMode = '2d';
    this.styleRevision = 0;
    this.innerHTML = `<main class="apartment" aria-label="尼古喵喵和朋友們的互動公寓" data-light="day" data-guest="none" data-style="2d">
      ${roomIllustration()}
      <div class="scene-three" aria-hidden="true"></div>
      <div class="light-wash" aria-hidden="true"></div>
      <div class="dust-motes" aria-hidden="true">${Array.from({length:12},(_,i)=>`<i style="--i:${i}"></i>`).join('')}</div>
      <div class="cast"></div>
      <character-picker></character-picker>
      <nav class="style-switch" aria-label="網頁風格"><button data-view-style="2d" aria-label="2D 動畫插畫風格" aria-pressed="true" title="2D 動畫插畫">2D</button><button data-view-style="3d" aria-label="3D 立體公寓風格" aria-pressed="false" title="3D 立體公寓">3D</button></nav>
      <div class="room-props">${PROP_LAYOUT.map(p=>`<room-prop data-id="${p.id}" data-label="${p.name}" style="--x:${p.x}%;--y:${p.y}%;--width:${p.width}vw;--mx:${p.mobile.x}%;--my:${p.mobile.y}%;--mwidth:${p.mobile.width}vw"></room-prop>`).join('')}</div>
      <div class="corner-controls"><button class="sound-toggle" aria-label="關閉音效" aria-pressed="false">${icon('volume')}</button><button class="reset-button" aria-label="重新開始">${icon('reset')}</button></div>
      <div class="first-touch hidden" aria-hidden="true">${icon('paw')}</div>
      <div class="three-actions" role="group" aria-label="3D 互動與視角" hidden>
        <button data-three-action="yani" aria-label="讓喵喵抽煙" title="讓喵喵抽煙">${icon('smoke')}</button>
        <button data-three-action="guest" aria-label="與朋友互動" title="與朋友互動" hidden>${icon('paw')}</button>
        <span class="action-divider" aria-hidden="true"></span>
        <button data-three-action="radio" aria-label="播放收音機，讓大家跳舞" title="一起跳舞">${icon('music')}</button>
        <button data-three-action="fan" aria-label="啟動電風扇" title="電風扇">${icon('wind')}</button>
        <button data-three-action="lamp" aria-label="切換白天與夜晚" title="切換日夜">${icon('sun')}</button>
        <span class="action-divider" aria-hidden="true"></span>
        <button data-three-action="camera" aria-label="重設 3D 視角" title="重設視角；拖曳畫面可旋轉">${icon('orbit')}</button>
      </div>
      <p class="style-notice" role="status" hidden></p>
      <p class="sr-only live-description" role="status" aria-live="polite">喵喵一直在家。從側邊邀請一位朋友，點兩人可以互相搗蛋。空白鍵讓喵喵抽煙，M 可靜音，R 可重置。</p>
    </main>`;
    this.stage = this.querySelector('.apartment');
    this.smokeEffect = new SmokeEffect(this.stage);
    this.gags = new GagEffects(this.stage);
    this.mountActor('yani');
    this.querySelector('character-picker').invited = null;

    this.onActorActivate = event => this.activateActor(event.detail);
    this.onPropActivate = event => this.activateProp(event.detail);
    this.onFriendChange = event => this.inviteFriend(event.detail);
    this.addEventListener('actor-activate', this.onActorActivate);
    this.addEventListener('prop-activate', this.onPropActivate);
    this.addEventListener('friend-change', this.onFriendChange);
    this.querySelector('.sound-toggle').addEventListener('click', () => this.toggleSound());
    this.querySelector('.reset-button').addEventListener('click', () => this.reset());
    this.querySelectorAll('[data-view-style]').forEach(button => button.addEventListener('click', () => this.setStyle(button.dataset.viewStyle)));
    this.querySelectorAll('[data-three-action]').forEach(button => button.addEventListener('click', () => {
      const action = button.dataset.threeAction;
      if(action==='camera') this.view3D?.resetCamera();
      else if(action==='yani') this.activateActor('yani');
      else if(action==='guest') { if(this.guestId) this.activateActor(this.guestId); }
      else this.activateProp(action);
    }));
    this.onKey = event => {
      if(event.repeat || /BUTTON|INPUT|TEXTAREA|SELECT|A/.test(document.activeElement.tagName) || document.activeElement.isContentEditable) return;
      if(event.code==='Space') { event.preventDefault(); this.activateActor('yani'); }
      if(event.code==='KeyM') this.toggleSound();
      if(event.code==='KeyR') this.reset();
    };
    document.addEventListener('keydown', this.onKey);
    this.resizeObserver = new ResizeObserver(() => this.reflowCast());
    this.resizeObserver.observe(this.stage);
    this.reflowCast();
    let preferredStyle = '2d';
    try { preferredStyle = localStorage.getItem('neko-style') || '2d'; } catch { /* Storage may be unavailable in private embeds. */ }
    if(preferredStyle==='3d') this.setStyle('3d');
  }

  async setStyle(mode) {
    if(!['2d','3d'].includes(mode)) return;
    const revision = ++this.styleRevision;
    this.clearActivity();
    this.showStyleNotice('');
    this.querySelector('[data-view-style="3d"]').removeAttribute('aria-busy');
    if(mode==='3d' && !this.view3D) {
      this.showStyleNotice('正在建立 3D 公寓…');
      this.querySelector('[data-view-style="3d"]').setAttribute('aria-busy','true');
      try {
        const { Room3D } = await import('./three/view3d.js');
        if(revision!==this.styleRevision || !this.isConnected) return;
        this.view3D = new Room3D(this.querySelector('.scene-three'), this);
      } catch {
        if(revision===this.styleRevision && this.isConnected) this.handle3DFailure();
        return;
      }
    }
    if(revision!==this.styleRevision || !this.isConnected) return;
    try {
      this.view3D?.syncActors(this.actors);
      this.applyStyle(mode);
      this.showStyleNotice('');
      this.announce(mode==='3d' ? '已切換 3D 公寓。拖曳旋轉、滾輪或雙指縮放；點角色互動，下方按鈕也能用鍵盤操作。' : '已切換 2D 插畫公寓，喵喵與來訪的朋友都還在。');
    } catch {
      this.handle3DFailure();
    }
  }

  applyStyle(mode) {
    const three = mode==='3d';
    this.styleMode = mode;
    this.stage.dataset.style = mode;
    this.querySelectorAll('[data-view-style]').forEach(button => {
      button.setAttribute('aria-pressed',String(button.dataset.viewStyle===mode));
      button.removeAttribute('aria-busy');
    });
    this.querySelector('.cast').inert = three;
    this.querySelector('.room-props').inert = three;
    this.querySelector('.scene-three').setAttribute('aria-hidden',String(!three));
    this.querySelector('.scene-three').inert = !three;
    this.querySelector('.three-actions').hidden = !three;
    this.updateGuestControl();
    this.view3D?.setActive(three);
    this.view3D?.resize();
    try { localStorage.setItem('neko-style',mode); } catch { /* The toggle also works without persistence. */ }
  }

  showStyleNotice(message) {
    clearTimeout(this.styleNoticeTimer);
    const notice = this.querySelector('.style-notice');
    notice.textContent = message;
    notice.hidden = !message;
  }

  handle3DFailure() {
    ++this.styleRevision;
    this.clearActivity();
    this.view3D?.dispose();
    this.view3D = null;
    this.querySelector('.scene-three').replaceChildren();
    this.applyStyle('2d');
    this.showStyleNotice('此裝置暫時無法顯示 3D，已保留 2D 公寓。');
    this.styleNoticeTimer = setTimeout(() => this.showStyleNotice(''),7000);
  }

  updateGuestControl() {
    const button = this.querySelector('[data-three-action="guest"]');
    button.hidden = !this.guestId;
    const label = this.guestId ? `與${this.actor(this.guestId).character.name}互動` : '與朋友互動';
    button.setAttribute('aria-label',label);
    button.title = label;
  }

  mountActor(id) {
    const definition = CHARACTER_ACTIONS.find(character => character.id===id);
    const actor = document.createElement('neko-actor');
    actor.dataset.id = id;
    actor.style.setProperty('--delay', `${definition.delay}s`);
    this.querySelector('.cast').append(actor);
    this.actors.set(id, actor);
    return actor;
  }

  inviteFriend(id) {
    const nextId = id==='none' || id===null ? null : id;
    if(nextId===this.guestId || (nextId && !CHARACTER_ACTIONS.some(c => c.id===nextId && c.id!=='yani'))) return;
    this.clearActivity();
    if(this.guestId) {
      this.actor(this.guestId).remove();
      this.actors.delete(this.guestId);
    }
    this.guestId = nextId;
    if(nextId) this.mountActor(nextId);
    this.stage.dataset.guest = nextId ?? 'none';
    this.querySelector('character-picker').invited = nextId;
    this.view3D?.syncActors(this.actors);
    this.updateGuestControl();
    this.reflowCast();
    if(nextId) {
      this.announce(`${this.actor(nextId).character.name}來作客了！點朋友與喵喵互相搗蛋，點喵喵看朋友對煙圈的反應。`);
      this.greetFriend(nextId, this.revision);
    } else this.announce('朋友先回家了，喵喵還在。點喵喵或按空白鍵讓牠抽煙。');
  }

  async greetFriend(id, revision) {
    await this.unlock();
    if(revision!==this.revision || this.busyUntil>performance.now()) return;
    this.actor('yani').react('laugh', 'goofy', 900);
    this.actor(id).react('wobble', 'normal', 900);
    this.gags.burst('notes', {...this.point(id,'head'), count:3});
    this.audio.play('meow');
  }

  reflowCast() {
    this.grounds = placeCast(this.stage, this.actors);
    this.view3D?.resize();
  }

  clearActivity() {
    this.revision++;
    this.timers.forEach(clearTimeout);
    this.timers.clear();
    this.lastActions.clear();
    this.busyUntil = 0;
    this.puffNumber = 0;
    this.smokeEffect.clear();
    this.gags.clear();
    this.actors.forEach(actor => actor.reset());
    this.stage.classList.remove('windy','dancing');
    const muted = this.audio.muted;
    this.audio.setMuted(true);
    this.audio.setMuted(muted);
  }

  later(callback, delay) {
    const timer = setTimeout(() => { this.timers.delete(timer); callback(); }, delay);
    this.timers.add(timer);
    return timer;
  }
  announce(text) { this.querySelector('.live-description').textContent = text; }
  actor(id) { return this.actors.get(id); }
  point(id, anchor) {
    return this.styleMode==='3d' && this.view3D ? this.view3D.point(id, anchor) : this.actor(id).point(this.stage, anchor);
  }
  ready(key, cooldown=750) {
    const now = performance.now();
    if(now-(this.lastActions.get(key)??-Infinity)<cooldown) return false;
    this.lastActions.set(key, now);
    return true;
  }
  async unlock() { await this.audio.unlock(); }

  async activateActor(id) {
    const definition = CHARACTER_ACTIONS.find(character => character.id===id);
    if(!this.actor(id) || !definition || this.busyUntil>performance.now()) return;
    // Give each pair's punchline time to finish before starting another scene.
    this.busyUntil = performance.now()+3600;
    const revision = this.revision;
    await this.unlock();
    if(revision!==this.revision) return;
    this.busyUntil = performance.now()+3600;
    CHARACTER_INTERACTIONS[definition.action](this);
  }

  async activateProp(id) {
    if(!this.ready(`prop-${id}`,1200)) return;
    const revision = this.revision;
    await this.unlock();
    if(revision!==this.revision) return;
    if(id==='lamp') {
      this.stage.dataset.light = this.stage.dataset.light==='day' ? 'night' : 'day';
      this.audio.play('switch');
      this.announce(this.stage.dataset.light==='night' ? '天黑了，吊燈亮了。' : '陽光照進房間。');
    }
    if(id==='fan') {
      this.stage.classList.add('windy');
      this.smokeEffect.clear();
      this.gags.wind();
      this.audio.play('fan');
      this.actors.forEach(actor => actor.react('wind','shock',2400));
      this.later(() => this.stage.classList.remove('windy'),2600);
      this.announce(this.guestId ? '電風扇開到最大，喵喵和朋友的頭髮都亂了。' : '電風扇開到最大，喵喵的頭髮都亂了。');
    }
    if(id==='radio') {
      this.stage.classList.add('dancing');
      this.audio.play('chaos');
      this.actors.forEach(actor => {
        actor.react('dance','goofy',3200);
        this.gags.burst('notes',{...this.point(actor.dataset.id,'head'),count:3});
      });
      this.later(() => this.audio.play('meow'),1500);
      this.later(() => this.stage.classList.remove('dancing'),3300);
      this.announce(this.guestId ? '收音機突然跑調，喵喵和朋友一起亂舞。' : '收音機突然跑調，喵喵開始亂舞。');
    }
  }

  async toggleSound() {
    this.audio.setMuted(!this.audio.muted);
    await this.audio.unlock();
    const button = this.querySelector('.sound-toggle');
    button.innerHTML = icon(this.audio.muted ? 'muted' : 'volume');
    button.setAttribute('aria-pressed', String(this.audio.muted));
    button.setAttribute('aria-label', this.audio.muted ? '開啟音效' : '關閉音效');
    if(!this.audio.muted) this.audio.play('meow');
  }
  reset() {
    this.clearActivity();
    this.stage.dataset.light = 'day';
    this.view3D?.resetCamera();
    this.announce(this.guestId ? '喵喵和朋友休息一下，公寓回到原來的樣子。' : '喵喵還在家，公寓回到原來的樣子。');
  }
  disconnectedCallback() {
    this.revision++;
    this.styleRevision++;
    clearTimeout(this.styleNoticeTimer);
    this.view3D?.dispose();
    this.view3D = null;
    this.timers.forEach(clearTimeout);
    this.timers.clear();
    this.resizeObserver?.disconnect();
    document.removeEventListener('keydown',this.onKey);
    this.removeEventListener('actor-activate',this.onActorActivate);
    this.removeEventListener('prop-activate',this.onPropActivate);
    this.removeEventListener('friend-change',this.onFriendChange);
    this.smokeEffect?.destroy();
    this.gags?.destroy();
    this.audio.dispose();
  }
}
customElements.define('neko-app', NekoApp);
