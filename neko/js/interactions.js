// Choreography uses shared actor anchors and effects, so either layout can play it.
const SMOKE_REACTIONS = {
  imouto: { action: 'bonk', message: '妹妹喵聞到煙味，拖鞋立刻鎖定喵喵！' },
  yaku: { action: 'sneeze', message: '煙圈飄到藥喵面前，糟了，鼻子開始癢了。' },
  kansai: { action: 'fan', message: '關西喵打開紙扇：這口煙，原封不動還給妳！' },
  aru: { action: 'hiccup', message: '酒喵把煙圈當成乾杯訊號，朝喵喵拋來罐子。' },
};

function later(app, delay, callback) {
  const revision = app.revision;
  return app.later(() => {
    if (app.revision === revision) callback();
  }, delay);
}

function react(app, id, motion, expression = 'goofy', duration = 1000) {
  const actor = app.actor(id);
  if (!actor) return false;
  actor.react(motion, expression, duration);
  return true;
}

function burst(app, id, kind, count = 4, anchor = 'head') {
  if (app.actor(id)) app.gags.burst(kind, { ...app.point(id, anchor), count });
}

function impact(app, id, { motion = 'bonk', expression = 'shock', sound = 'bonk', count = 5, duration = 1000 } = {}) {
  if (!react(app, id, motion, expression, duration)) return;
  burst(app, id, 'stars', count);
  app.audio.play(sound);
}

function pairPresent(app, guest) {
  return Boolean(app.actor('yani') && app.actor(guest));
}

async function toss(app, kind, sender, recipient, { duration = 580, target = 'head' } = {}) {
  const fromActor = app.actor(sender);
  const toActor = app.actor(recipient);
  if (!fromActor || !toActor) return false;
  const revision = app.revision;
  await app.gags.projectile(kind, app.point(sender, 'hand'), app.point(recipient, target), { duration });
  return app.revision === revision && app.actor(sender) === fromActor && app.actor(recipient) === toActor;
}

function sneezeTrail(app, sender, recipient) {
  if (!app.actor(sender) || !app.actor(recipient)) return;
  const from = app.point(sender, 'mouth');
  const to = app.point(recipient, 'head');
  [0.2, 0.55, 0.9].forEach((fraction, index) => {
    later(app, index * 75, () => {
      if (!app.actor(sender) || !app.actor(recipient)) return;
      app.gags.burst('drops', {
        x: from.x + (to.x - from.x) * fraction,
        y: from.y + (to.y - from.y) * fraction,
        count: 2,
      });
    });
  });
}

function smoke(app) {
  const yani = app.actor('yani');
  if (!yani) return;
  const count = ++app.puffNumber;
  const guest = app.guestId;
  yani.smoke(3100);
  app.audio.play('lighter');
  app.announce('喵喵點燃香菸，得意地準備吐煙圈。');

  later(app, 400, () => {
    if (!app.actor('yani')) return;
    app.audio.play('puff');
    app.smokeEffect.puff({ ...app.point('yani', 'tip'), intensity: 1.1 });
  });
  later(app, 850, () => {
    if (app.actor('yani')) app.gags.smokeRing({ ...app.point('yani', 'mouth'), fish: count % 2 === 0 });
  });

  const reaction = SMOKE_REACTIONS[guest];
  if (reaction) {
    later(app, 1050, () => {
      if (app.guestId !== guest || !pairPresent(app, guest)) return;
      CHARACTER_INTERACTIONS[reaction.action](app);
      app.announce(reaction.message);
    });
  }

  if (count % 3 === 0) {
    later(app, reaction ? 3000 : 1800, () => {
      if (!react(app, 'yani', 'sneeze', 'goofy', reaction ? 550 : 1200)) return;
      burst(app, 'yani', 'drops', 4, 'mouth');
      app.audio.play('cough');
      app.announce('第三口煙，喵喵還是被自己嗆到了。');
    });
  }
}

async function bonk(app) {
  if (!pairPresent(app, 'imouto')) return;
  react(app, 'imouto', 'throw', 'annoyed', 1400);
  burst(app, 'imouto', 'anger', 3);
  app.audio.play('gasp');
  app.announce('妹妹喵的拖鞋朝喵喵飛過去了！');
  if (!await toss(app, 'slipper', 'imouto', 'yani', { duration: 580 })) return;

  impact(app, 'yani', { count: 6, duration: 950 });
  react(app, 'imouto', 'laugh', 'goofy', 1200);
  app.announce('啪！拖鞋命中喵喵，妹妹喵笑到耳朵都在抖。');
  later(app, 280, () => {
    if (!pairPresent(app, 'imouto')) return;
    react(app, 'yani', 'wobble', 'goofy', 900);
    app.audio.play('boing');
    burst(app, 'yani', 'stars', 3);
  });
}

function sneeze(app) {
  if (!pairPresent(app, 'yaku')) return;
  react(app, 'yaku', 'windup', 'shock', 470);
  app.audio.play('gasp');
  app.announce('藥喵吸一口氣……喵喵還沒發現大事不妙。');

  later(app, 450, () => {
    if (!pairPresent(app, 'yaku')) return;
    react(app, 'yaku', 'sneeze', 'goofy', 950);
    app.audio.play('cough');
    burst(app, 'yaku', 'drops', 6, 'mouth');
    sneezeTrail(app, 'yaku', 'yani');
  });
  later(app, 630, () => {
    if (!pairPresent(app, 'yaku')) return;
    react(app, 'yani', 'wind', 'shock', 780);
    burst(app, 'yani', 'drops', 4);
    app.audio.play('boing');
    app.announce('哈啾！藥喵的噴嚏把喵喵的瀏海吹歪了。');
  });
  later(app, 970, () => {
    if (!pairPresent(app, 'yaku')) return;
    react(app, 'yani', 'sneeze', 'goofy', 850);
    app.audio.play('cough');
    burst(app, 'yani', 'drops', 4, 'mouth');
    react(app, 'yaku', 'laugh', 'goofy', 900);
    app.announce('喵喵也跟著哈啾！藥喵忍不住笑了。');
  });
}

function fan(app) {
  if (!pairPresent(app, 'kansai')) return;
  react(app, 'kansai', 'throw', 'annoyed', 1000);
  react(app, 'yani', 'wind', 'shock', 900);
  app.smokeEffect.clear();
  app.gags.wind();
  app.audio.play('fan');
  const from = app.point('kansai', 'hand');
  const to = app.point('yani', 'mouth');
  app.smokeEffect.puff({ x: (from.x + to.x) / 2, y: (from.y + to.y) / 2, intensity: 0.5 });
  app.announce('關西喵把煙搧回去，喵喵的頭髮全亂了。');

  later(app, 170, () => {
    if (pairPresent(app, 'kansai')) app.smokeEffect.puff({ ...app.point('yani', 'mouth'), intensity: 0.65 });
  });
  later(app, 420, async () => {
    if (!pairPresent(app, 'kansai')) return;
    if (!await toss(app, 'fan', 'kansai', 'yani', { duration: 480 })) return;
    impact(app, 'yani', { expression: 'goofy', count: 4, duration: 900 });
    react(app, 'kansai', 'laugh', 'goofy', 1150);
    app.announce('啪嗒！紙扇輕敲喵喵的頭，關西喵笑得停不下來。');
    later(app, 300, () => {
      if (!pairPresent(app, 'kansai')) return;
      react(app, 'yani', 'laugh', 'goofy', 800);
      app.audio.play('meow');
    });
  });
}

function hiccup(app) {
  if (!pairPresent(app, 'aru')) return;
  react(app, 'aru', 'hiccup', 'goofy', 950);
  app.audio.play('hiccup');
  burst(app, 'aru', 'bubbles', 4, 'mouth');
  app.announce('酒喵一個嗝，把手上的罐子拋給喵喵。');

  later(app, 250, async () => {
    if (!pairPresent(app, 'aru')) return;
    react(app, 'aru', 'throw', 'goofy', 700);
    app.audio.play('can');
    if (!await toss(app, 'can', 'aru', 'yani', { duration: 620, target: 'hand' })) return;

    react(app, 'yani', 'jump', 'shock', 550);
    burst(app, 'yani', 'stars', 3, 'hand');
    app.audio.play('can');
    app.announce('喵喵手忙腳亂接住罐子，居然也開始打嗝！');
    later(app, 240, () => {
      if (!pairPresent(app, 'aru')) return;
      react(app, 'yani', 'hiccup', 'goofy', 900);
      react(app, 'aru', 'laugh', 'goofy', 900);
      burst(app, 'yani', 'bubbles', 4, 'mouth');
      app.audio.play('hiccup');
    });
    later(app, 610, () => {
      if (!pairPresent(app, 'aru')) return;
      react(app, 'aru', 'hiccup', 'goofy', 800);
      react(app, 'yani', 'laugh', 'goofy', 800);
      burst(app, 'aru', 'bubbles', 3, 'mouth');
      app.audio.play('hiccup');
      app.announce('嗝、嗝！兩隻喵輪流打嗝，最後一起笑成一團。');
    });
  });
}

export const CHARACTER_INTERACTIONS = Object.freeze({ smoke, bonk, sneeze, fan, hiccup });
