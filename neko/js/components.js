import { CHARACTERS, renderCharacter } from './characters.js';
import { renderProp } from './scene.js';

// One actor implementation for every character: rendering, keyboard activation,
// expression changes, timed reactions, and anchored visual effects.
export class NekoActor extends HTMLElement {
  connectedCallback() {
    const id = this.dataset.id;
    const character = CHARACTERS.find(item => item.id === id);
    if (!character) return;
    this.character = character;
    this.dataset.pose = id === 'yani' ? 'sit' : 'stand';
    this.innerHTML = `<button class="actor-hit" aria-label="${character.name}，點擊互動">${renderCharacter(character, { pose:this.dataset.pose, instanceId:`scene-${id}` })}</button>${this.dataset.pose==='sit'?'<span class="seat-cushion" aria-hidden="true"></span>':''}<span class="ground-shadow" aria-hidden="true"></span>`;
    this.querySelector('button').addEventListener('click', () => this.dispatchEvent(new CustomEvent('actor-activate',{detail:id,bubbles:true})));
  }
  react(motion, expression = 'normal', duration = 1600) {
    clearTimeout(this.reactionTimer);
    this.dataset.motion = motion;
    this.dataset.reaction = expression;
    this.reactionTimer = setTimeout(() => { delete this.dataset.motion; delete this.dataset.reaction; }, duration);
  }
  smoke(duration = 3300) {
    clearTimeout(this.smokeTimer);
    this.dataset.smoking = 'true';
    this.react('smoke','normal',duration);
    this.smokeTimer = setTimeout(() => delete this.dataset.smoking, duration);
  }
  point(stage, point='mouth') {
    const bounds = this.querySelector('svg').getBoundingClientRect();
    const scene = stage.getBoundingClientRect();
    const points = { mouth:[120,119], tip:[168,130], head:[120,47], hand:[145,173], center:[120,210] };
    const [x,y] = points[point] || points.mouth;
    return { x:(bounds.left-scene.left+bounds.width*x/240)/scene.width*100, y:(bounds.top-scene.top+bounds.height*y/420)/scene.height*100 };
  }
  reset() { clearTimeout(this.reactionTimer); clearTimeout(this.smokeTimer); delete this.dataset.motion; delete this.dataset.reaction; delete this.dataset.smoking; }
  disconnectedCallback() { this.reset(); }
}

export class RoomProp extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `<button class="prop-hit" aria-label="${this.dataset.label}">${renderProp(this.dataset.id)}</button>`;
    this.querySelector('button').addEventListener('click', () => this.dispatchEvent(new CustomEvent('prop-activate',{detail:this.dataset.id,bubbles:true})));
  }
}

// Portraits reuse the same vector renderer as the full actor.
export class CharacterPicker extends HTMLElement {
  connectedCallback() {
    this.innerHTML = `<nav class="character-picker" aria-label="邀請一位朋友陪喵喵">
      <button class="picker-option picker-empty" data-invite-character="none" aria-label="請朋友離開，喵喵留下" aria-pressed="false" title="喵喵自己待著"><svg viewBox="0 0 32 32" aria-hidden="true"><path d="M15 7H8v18h7m-3-9h14m-5-5 5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg></button>
      ${CHARACTERS.filter(character=>character.id!=='yani').map(character=>`<button class="picker-option" data-invite-character="${character.id}" aria-label="邀請${character.name}" aria-pressed="false" title="邀請${character.name}"><span class="picker-portrait">${renderCharacter(character,{portrait:true,instanceId:`picker-${character.id}`})}</span></button>`).join('')}
    </nav>`;
    this.querySelectorAll('[data-invite-character]').forEach(button=>button.addEventListener('click',()=>this.dispatchEvent(new CustomEvent('friend-change',{detail:button.dataset.inviteCharacter,bubbles:true}))));
    this.invited = this._invited ?? null;
  }
  set invited(id) { this._invited = id && id !== 'none' ? id : null; this.querySelectorAll('[data-invite-character]').forEach(button=>button.setAttribute('aria-pressed', String(button.dataset.inviteCharacter===(this._invited || 'none')))); }
  get invited() { return this._invited; }
}

customElements.define('neko-actor', NekoActor);
customElements.define('room-prop', RoomProp);
customElements.define('character-picker', CharacterPicker);
