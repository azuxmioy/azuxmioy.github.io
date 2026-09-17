import * as THREE from '../../assets/vendor/three/three.module.min.js';
import { CHARACTERS } from '../characters.js';
import { createTextureKit } from './textures3d.js';
import { loft, sweep, combine, tailoredPanel, sculptedHead, sculptedEar, scalp } from './geometry3d.js';

// Shared proportions keep the skull, hair, face UVs and effect anchors together.
// Seated height comes from bent limbs, rather than compressing the whole body.
const ANATOMY = Object.freeze({
  head: [0.92, 1.05, 1.20],
  torso: [1.08, 1.20, 1.14],
  arm: [1.08, 1.16, 1.16],
  seated: { waist: 0.65, head: 2.38 },
  standing: { waist: 1.31, head: 3.04 },
});

/** Original, UV-unwrapped character sculptures sharing one anatomy/wardrobe kit. */
export function createCharacter3D(id) {
  const character = CHARACTERS.find(item => item.id === id);
  if (!character) throw new TypeError(`Unknown character: ${id}`);
  const a = character.appearance, seated = id === 'yani';
  const kit = createTextureKit(a, id);
  const group = new THREE.Group();
  group.name = id;
  group.userData.actorId = id;
  const pose = new THREE.Group();
  group.add(pose);
  const geometries = new Set();
  const skin = kit.material('skin', a.skin);
  const hair = kit.material('hair', a.hair);
  const hairShade = kit.material('hair', a.hairShade);
  const clothKind = ['ribbed', 'sweater'].includes(a.outfit) ? 'knit' : 'cloth';
  const cloth = kit.material(clothKind, a.shirt);
  const clothShade = kit.material(clothKind, a.shirtShade);
  const trouser = kit.material(seated ? 'denim' : 'cloth', a.pants);
  const proportions = seated ? ANATOMY.seated : ANATOMY.standing;
  const bodyBase = proportions.waist;
  const headHeight = proportions.head;

  function mesh(parent, part, geometry, material) {
    const object = new THREE.Mesh(geometry, material);
    object.name = `${id}-${part}`;
    object.userData.actorId = id;
    object.userData.part = part;
    object.castShadow = true;
    object.receiveShadow = true;
    geometries.add(geometry);
    parent.add(object);
    return object;
  }
  function anchor(parent, name, point) {
    const object = new THREE.Object3D();
    object.name = `${id}-${name}`;
    object.position.set(...point);
    parent.add(object);
    return object;
  }
  function seam(parent, part, points, width, material) {
    return mesh(parent, part, sweep(points, [width, width, width], { radial: 6, rows: 20 }), material);
  }
  function cuff(parent, part, position, radius, thickness, material, depth = 0.75) {
    const geometry = new THREE.TorusGeometry(radius, thickness, 8, 40);
    geometry.rotateX(Math.PI / 2);
    geometry.scale(1, 1, depth);
    const object = mesh(parent, part, geometry, material);
    object.position.set(...position);
    return object;
  }
  const clothSculpt = (amount = 0.007) => ([x, y, z], phi, v) => {
    const wrinkles = amount * Math.sin(phi * 9 + y * 3) * Math.sin(Math.PI * v) +
      amount * 0.7 * Math.sin(y * 40 + phi * 2) * Math.exp(-(((v - 0.18) / 0.13) ** 2));
    return [x + Math.sin(phi) * wrinkles, y, z + Math.cos(phi) * wrinkles];
  };

  if (seated) {
    mesh(group, 'cushion', loft([[0, 0.75, 0.47], [0.025, 0.90, 0.59], [0.08, 0.99, 0.65],
      [0.15, 0.96, 0.63], [0.21, 0.76, 0.49], [0.23, 0, 0]], {
      rows: 28, sculpt: ([x, y, z], phi) => [x, y - Math.max(0, y - 0.10) * 0.32 * Math.sin(phi * 2) ** 2, z + 0.04],
    }), kit.material('cloth', '#89936b'));
    cuff(group, 'cushion-piping', [0, 0.105, 0.04], 0.967, 0.008, kit.material('cloth', '#b3ba8b'), 0.66);
    // A rounded pelvis and two bent trouser legs form a real crossed-leg pose.
    // The offset at the crossing preserves each shin's silhouette from the side.
    const pelvis = loft([[0.28, 0, 0], [0.38, 0.38, 0.28], [0.52, 0.47, 0.35],
      [0.68, 0.43, 0.33], [0.79, 0.39, 0.28]], { rows: 28, sculpt: clothSculpt(0.01) });
    pelvis.translate(0, 0, -0.07);
    const crossedLegs = [-1, 1].map(side => sweep([
      [side * 0.25, 0.62, -0.05], [side * 0.57, 0.51, 0.17],
      [side * 0.64, 0.40, 0.40], [side * 0.20, side < 0 ? 0.40 : 0.29, side < 0 ? 0.62 : 0.78],
      [-side * 0.36, side < 0 ? 0.39 : 0.28, side < 0 ? 0.74 : 0.85],
    ], [0.24, 0.255, 0.25, 0.185, 0.14], {
      depths: [0.235, 0.245, 0.23, 0.17, 0.13], radial: 32, rows: 52, folds: 0.035,
    }));
    mesh(pose, 'pants', combine([pelvis, ...crossedLegs]), trouser);
    for (const side of [-1, 1]) {
      const foot = mesh(pose, 'bare-foot', loft([[0.145, 0, 0], [0.175, 0.14, 0.21],
        [0.24, 0.175, 0.235], [0.315, 0.14, 0.19], [0.36, 0.075, 0.105], [0.375, 0, 0]], {
        rows: 22, radial: 32, sculpt: ([x, y, z], phi) => {
          const toes = Math.max(0, Math.cos(phi)) ** 8 * Math.sin(phi * 17) * 0.009;
          return [x, y, z + toes];
        },
      }), skin);
      foot.position.set(side * 0.46, side > 0 ? 0.11 : 0, side > 0 ? 0.78 : 0.92);
      foot.rotation.y = side * 0.80;
    }
  } else {
    const skirt = ['uniform', 'ribbed'].includes(a.outfit);
    const legGeometries = [];
    for (const side of [-1, 1]) {
      const x = side * 0.215;
      legGeometries.push(skirt ? sweep([[x, 1.42, 0], [x, 0.95, 0.02], [x + side * 0.01, 0.55, 0.043]],
        [0.155, 0.15, 0.135], { depths: [0.17, 0.15, 0.127], radial: 20, rows: 28 }) :
        sweep([[x, 1.42, 0], [x, 0.95, 0.02], [x + side * 0.01, 0.48, 0.05],
          [x + side * 0.015, 0.18, 0.055]], [0.155, 0.15, 0.13, 0.115], {
          depths: [0.17, 0.15, 0.12, 0.105], radial: 20, rows: 34, folds: 0.035,
        }));
      if (skirt) mesh(pose, 'stocking', sweep([[x + side * 0.01, 0.60, 0.04], [x + side * 0.012, 0.37, 0.05],
        [x + side * 0.015, 0.15, 0.055]], [0.15, 0.133, 0.12], { depths: [0.145, 0.125, 0.112], radial: 20, rows: 24 }),
      kit.material('knit', a.outfit === 'uniform' ? '#343440' : '#ebe1ca'));
      const shoeX = x + side * 0.025;
      const shoeColor = a.outfit === 'uniform' ? '#635244' : '#d9ceb5';
      const sole = mesh(pose, 'shoe-sole', loft([[0, 0.15, 0.22], [0.018, 0.191, 0.295],
        [0.06, 0.192, 0.298], [0.072, 0.17, 0.27]], { rows: 10, radial: 32 }), kit.material('leather', '#665e51'));
      sole.position.set(shoeX, 0, 0.14);
      const shoe = mesh(pose, 'shoe-upper', loft([[0.054, 0.18, 0.285], [0.105, 0.187, 0.29],
        [0.165, 0.159, 0.244], [0.205, 0.12, 0.16], [0.22, 0.085, 0.10]], {
        rows: 24, radial: 36, sculpt: ([px, y, pz], phi) => [px, y + Math.max(0, -Math.cos(phi)) * 0.015, pz],
      }), kit.material('leather', shoeColor));
      shoe.position.set(shoeX, 0, 0.14);
      seam(pose, 'shoe-stitch', [[shoeX - 0.125, 0.136, 0.32], [shoeX, 0.17, 0.385],
        [shoeX + 0.125, 0.136, 0.32]], 0.006, kit.material('leather', '#ac9e85'));
    }
    mesh(pose, skirt ? 'legs' : 'pants', combine(legGeometries), skirt ? skin : trouser);
  }

  const torso = new THREE.Group();
  torso.position.y = bodyBase;
  torso.scale.set(...ANATOMY.torso);
  pose.add(torso);
  mesh(torso, 'shirt', loft([[0.018, 0.405, 0.28], [0.052, 0.445, 0.31], [0.16, 0.448, 0.31],
    [0.36, 0.422, 0.30], [0.59, 0.387, 0.27], [0.73, 0.354, 0.238],
    [0.81, 0.27, 0.19], [0.835, 0.17, 0.139]], { rows: 44, sculpt: clothSculpt() }), cloth);
  cuff(torso, 'shirt-hem', [0, 0.065, 0], 0.436, 0.009, clothShade, 0.695);
  mesh(torso, 'neck', loft([[0.76, 0.14, 0.13], [0.91, 0.145, 0.13], [1.00, 0.19, 0.145]], { rows: 12, radial: 24 }), skin);
  cuff(torso, 'collar', [0, 0.828, 0], 0.174, 0.022, clothShade, 0.83);
  if (['uniform', 'ribbed'].includes(a.outfit)) {
    mesh(torso, 'pleated-skirt', loft([[-0.285, 0.535, 0.36], [-0.22, 0.515, 0.35],
      [0.08, 0.405, 0.28]], { rows: 26, radial: 64, sculpt: ([x, y, z], phi) => {
      const pleat = 0.018 * Math.cos(phi * 14) * (1 - (y + 0.285) / 0.365);
      return [x + Math.sin(phi) * pleat, y, z + Math.cos(phi) * pleat];
    } }), trouser);
  }
  if (a.outfit === 'uniform') {
    const lapels = [], ribbons = [];
    for (const side of [-1, 1]) {
      const lapel = tailoredPanel([[0, 0.49], [side * 0.25, 0.79], [side * 0.29, 0.65], [side * 0.13, 0.43]], 0.015);
      lapel.translate(0, 0, 0.27);
      lapels.push(lapel);
      const ribbon = tailoredPanel([[0, 0.57], [side * 0.17, 0.66], [side * 0.16, 0.48], [0, 0.54]], 0.022, 0.008);
      ribbon.translate(0, 0, 0.31);
      ribbons.push(ribbon);
    }
    mesh(torso, 'uniform-lapels', combine(lapels), kit.material('cloth', '#ece4cf'));
    mesh(torso, 'uniform-bow', combine(ribbons), kit.material('cloth', a.accent));
    seam(torso, 'placket', [[0, 0.50, 0.314], [0, 0.3, 0.32], [0, 0.095, 0.322]], 0.007, clothShade);
  } else if (a.outfit === 'ribbed') {
    mesh(torso, 'knitted-collar', loft([[0.79, 0.19, 0.15], [0.86, 0.19, 0.15], [0.9, 0.175, 0.142]], {
      rows: 12, radial: 48, sculpt: ([x, y, z], phi) => {
        const rib = 0.004 * Math.sin(phi * 24);
        return [x + Math.sin(phi) * rib, y, z + Math.cos(phi) * rib];
      },
    }), cloth);
  } else if (a.outfit === 'hoodie') {
    mesh(torso, 'hood', loft([[0.60, 0.215, 0.15], [0.68, 0.30, 0.235],
      [0.77, 0.28, 0.25], [0.845, 0.20, 0.16]], { rows: 26, radial: 40, sculpt: ([x, y, z], phi) =>
      [x, y + 0.075 * Math.max(0, -Math.cos(phi)), z - 0.03] }), kit.material('cloth', a.accent));
    const pocket = mesh(torso, 'hoodie-pocket', tailoredPanel([[-0.255, 0.18], [-0.27, 0.31],
      [-0.14, 0.37], [0.14, 0.37], [0.27, 0.31], [0.255, 0.18]], 0.009, 0.008), clothShade);
    pocket.position.z = 0.297;
    const cords = [];
    for (const side of [-1, 1]) cords.push(sweep([[side * 0.14, 0.75, 0.24],
      [side * 0.18, 0.56, 0.3], [side * 0.16, 0.43, 0.32]], [0.009, 0.009], { radial: 6, rows: 16 }));
    mesh(torso, 'hoodie-cords', combine(cords), kit.material('cloth', '#826b51'));
  }

  const arms = {};
  function makeArm(side) {
    const arm = new THREE.Group();
    arm.position.set(side * 0.32 * ANATOMY.torso[0], bodyBase + 0.71 * ANATOMY.torso[1], 0);
    arm.scale.set(...ANATOMY.arm);
    pose.add(arm);
    const raised = seated && side === 1;
    const root = [-side * 0.10, -0.025, 0];
    const shoulder = [side * 0.055, -0.065, 0.015];
    const elbow = [side * 0.27, -0.29, 0.07];
    const wrist = raised ? [side * 0.07, 0.27, 0.43] : [side * 0.22, -0.56, 0.16];
    if (a.outfit === 'tee') {
      mesh(arm, 'sleeve', sweep([root, shoulder, [side * 0.17, -0.14, 0.04], elbow],
        [0.12, 0.176, 0.17, 0.136], { depths: [0.105, 0.157, 0.155, 0.12], radial: 24, rows: 30, folds: 0.03 }), cloth);
      mesh(arm, 'forearm', sweep([elbow, [side * (raised ? 0.18 : 0.235), raised ? -0.12 : -0.43, raised ? 0.20 : 0.11], wrist],
        [0.105, 0.085, 0.062], { depths: [0.095, 0.078, 0.058], radial: 20, rows: 30 }), skin);
    } else {
      mesh(arm, 'sleeve', sweep([root, shoulder, [side * 0.18, -0.19, 0.025], elbow, wrist],
        [0.12, 0.172, 0.165, 0.14, 0.115], { depths: [0.105, 0.153, 0.15, 0.13, 0.11], radial: 24, rows: 36, folds: 0.035 }), cloth);
    }
    const hand = new THREE.Group();
    hand.position.set(...wrist);
    arm.add(hand);
    const palm = loft([[-0.14, 0, 0], [-0.12, 0.064, 0.046], [-0.065, 0.086, 0.058],
      [0.005, 0.069, 0.054], [0.055, 0.058, 0.047]], { rows: 20, radial: 24 });
    const fingers = [-0.052, -0.015, 0.024, 0.058].map((x, index) => {
      const length = [0.087, 0.115, 0.103, 0.073][index];
      return sweep([[x, -0.092, 0], [x * 1.13, -0.137, 0.006],
        [x * 1.16, -0.10 - length, 0.023]], [0.024, 0.023, 0.008], {
        depths: [0.030, 0.025, 0.009], rows: 16, radial: 12,
      });
    });
    const thumb = sweep([[-0.06, -0.013, 0], [-0.105, -0.049, 0.015],
      [-0.112, -0.098, 0.041]], [0.034, 0.030, 0.012], { radial: 16, rows: 18 });
    const geometry = combine([palm, thumb, ...fingers]);
    if (raised) { geometry.rotateZ(Math.PI); geometry.scale(0.88, 1.04, 1); }
    if (side < 0) geometry.rotateY(Math.PI);
    mesh(hand, 'hand', geometry, skin);
    arms[side] = arm;
    return hand;
  }
  makeArm(-1);
  const rightHand = makeArm(1);

  const tail = new THREE.Group();
  tail.position.set(0.39, seated ? 0.45 : 1.12, -0.20);
  pose.add(tail);
  mesh(tail, 'tail', sweep([[0, 0, 0], [0.46, -0.05, 0], [0.73, 0.10, 0.04],
    [0.80, 0.42, 0.07], [0.75, 0.57, 0.1]], [0.105, 0.095, 0.093, 0.075, 0.003],
  { radial: 18, rows: 48 }), hair);

  const head = new THREE.Group();
  head.position.set(0, headHeight, 0.015);
  head.scale.set(...ANATOMY.head);
  pose.add(head);
  const faceMesh = mesh(head, 'head', sculptedHead(), kit.face('normal', false));
  faceMesh.userData.expression = 'normal';
  mesh(head, 'hair-scalp', scalp(), hair);
  const ears = [];
  for (const side of [-1, 1]) {
    const ear = new THREE.Group();
    ear.position.set(side * 0.49, 0.42, -0.07);
    ear.rotation.z = -side * 0.22;
    head.add(ear);
    const earGeometry = sculptedEar();
    mesh(ear, 'cat-ear', earGeometry.shell, hair);
    mesh(ear, 'ear-lining', earGeometry.lining, kit.material('skin', '#cc9591'));
    ears.push(ear);
  }

  const fringe = new THREE.Group();
  head.add(fringe);
  const straight = a.hairstyle === 'straight';
  const long = ['long', 'curly'].includes(a.hairstyle);
  const frontLocks = [], sideLocks = [], backLocks = [];
  function hairLock(points, width, { rows = 16, taper = 0.02 } = {}) {
    return sweep(points, [width * 0.7, width, width * 0.62, taper * width], {
      depths: [0.040, 0.068, 0.045, 0.003], radial: 12, rows, ribbon: true,
    });
  }
  for (let i = -3; i <= 3; i++) {
    const x = i * 0.174;
    const lean = ['long', 'curly'].includes(a.hairstyle) ? 0.12 : (i % 2 ? 0.052 : -0.028);
    const tip = straight ? 0.125 + (i === 0 ? 0.02 : 0) : 0.062 + (i % 2 ? 0.115 : 0);
    frontLocks.push(hairLock([[x * 0.73, 0.57, 0.32], [x * 0.94, 0.37, 0.49],
      [x + lean * 0.65, 0.21, 0.55], [x + lean, tip, 0.545]], straight ? 0.115 : 0.141));
  }
  for (const side of [-1, 1]) {
    sideLocks.push(hairLock([[side * 0.52, 0.43, 0.26], [side * 0.655, 0.15, 0.285],
      [side * 0.65, -0.14, 0.30], [side * 0.615, -0.40, 0.26]], 0.17, { rows: 24 }));
    if (['shag', 'bob'].includes(a.hairstyle)) {
      for (let i = 0; i < 3; i++) sideLocks.push(hairLock([[side * 0.57, 0.08 - i * 0.10, -0.1],
        [side * 0.7, -0.11 - i * 0.11, -0.015], [side * (0.81 - i * 0.025), -0.16 - i * 0.14, 0.03]], 0.125));
    }
    if (long) {
      for (let i = 0; i < 3; i++) {
        const x = side * (0.44 + i * 0.105);
        const bend = a.hairstyle === 'curly' ? side * 0.085 : -side * 0.045;
        backLocks.push(hairLock([[x, 0.18, -0.23], [x + bend, -0.37, -0.28],
          [x - bend, -0.84, -0.20], [x + bend, -1.29 + i * 0.035, -0.07]], 0.185, { rows: 30 }));
      }
    }
  }
  mesh(fringe, 'hair-fringe', combine(frontLocks), hair);
  mesh(head, 'hair-sides', combine(sideLocks), hair);
  if (!long) {
    for (let i = -3; i <= 3; i++) {
      const phi = Math.PI + i * 0.24;
      backLocks.push(hairLock([
        [Math.sin(phi) * 0.57, 0.18, Math.cos(phi) * 0.50 - 0.04],
        [Math.sin(phi) * 0.64, -0.12, Math.cos(phi) * 0.49 - 0.04],
        [Math.sin(phi) * 0.55, -0.36, Math.cos(phi) * 0.40 - 0.04],
        [Math.sin(phi) * 0.47, -0.55 + (i % 2) * 0.025, Math.cos(phi) * 0.29 - 0.04],
      ], 0.13, { rows: 22 }));
    }
    mesh(head, 'hair-nape', combine(backLocks), hairShade);
  }
  if (long) mesh(head, 'hair-lengths', combine(backLocks), hairShade);
  if (a.hairstyle === 'curly') mesh(head, 'hair-curl', sweep([[-0.06, 0.62, -0.02], [-0.24, 0.86, 0],
    [-0.07, 0.94, 0], [0.12, 0.79, 0], [0.07, 0.69, 0]], [0.06, 0.063, 0.057, 0.045, 0.005],
  { depths: [0.025, 0.036, 0.029, 0.022, 0.002], radial: 8, rows: 32, ribbon: true }), hair);

  const cigarette = new THREE.Group();
  head.add(cigarette);
  const cigaretteStart = [0.035, -0.285, 0.545], filterEnd = [0.15, -0.30, 0.615], cigaretteEnd = [0.51, -0.345, 0.83];
  mesh(cigarette, 'cigarette-filter', sweep([cigaretteStart, filterEnd], [0.024, 0.024], { radial: 12, rows: 4 }), kit.material('cloth', '#c99661'));
  mesh(cigarette, 'cigarette-paper', sweep([filterEnd, cigaretteEnd], [0.023, 0.023], { radial: 12, rows: 4 }), kit.material('cloth', '#f6f1de'));
  const ember = mesh(cigarette, 'cigarette-ember', sweep([[0.485, -0.342, 0.815], cigaretteEnd], [0.024, 0.022],
    { radial: 12, rows: 3 }), kit.material('cloth', '#d46532', { emissive: '#f15b16', emissiveIntensity: 0.7 }));
  cigarette.visible = false;
  const anchors = {
    mouth: anchor(head, 'mouth', [0.02, -0.28, 0.61]),
    tip: anchor(head, 'tip', cigaretteEnd),
    head: anchor(head, 'head', [0, 0.55, 0.15]),
    hand: anchor(rightHand, 'hand', [0, 0.04, 0.06]),
    center: anchor(torso, 'center', [0, 0.4, 0.25]),
  };
  let previousMotion = '', motionStart = 0, disposed = false;
  const phase = CHARACTERS.findIndex(item => item.id === id) * 1.47;
  const validExpressions = new Set(['normal', 'shock', 'annoyed', 'goofy']);
  function update(state = {}, time = 0, reducedMotion = false) {
    if (disposed) return;
    const motion = state.motion || '';
    if (motion !== previousMotion) { previousMotion = motion; motionStart = time; }
    const elapsed = Math.max(0, time - motionStart);
    const expression = validExpressions.has(state.reaction) ? state.reaction : 'normal';
    const blinkPhase = (time + phase) % 5.7;
    const blinking = !reducedMotion && blinkPhase > 5.48 && ['normal', 'annoyed'].includes(expression);
    faceMesh.material = kit.face(expression, blinking);
    faceMesh.userData.expression = expression;
    faceMesh.userData.blink = blinking;
    cigarette.visible = state.smoking === 'true' || state.smoking === true;
    ember.material.emissiveIntensity = cigarette.visible ? 0.6 + Math.sin(time * 7) * 0.25 : 0;
    pose.position.y = 0;
    pose.rotation.set(0, 0, 0);
    pose.scale.set(1, 1, 1);
    head.rotation.set(0, 0, 0);
    torso.rotation.set(0, 0, 0);
    torso.scale.y = ANATOMY.torso[1];
    arms[-1].rotation.set(0, 0, 0);
    arms[1].rotation.set(0, 0, 0);
    fringe.rotation.z = 0;
    tail.rotation.set(0, 0, 0);
    ears.forEach((ear, index) => { ear.rotation.z = (index === 0 ? 1 : -1) * 0.22; });
    if (reducedMotion) return;
    tail.rotation.y = Math.sin(time * 1.5 + phase) * 0.10;
    head.rotation.z = Math.sin(time * 0.8 + phase) * 0.012;
    const oscillation = Math.sin(elapsed * 16);
    if (cigarette.visible) {
      head.rotation.x = -0.045 + Math.sin(time * 3) * 0.014;
      arms[1].rotation.z = Math.sin(time * 3) * 0.025;
    }
    switch (motion) {
      case 'throw':
        arms[1].rotation.z = -0.7 - Math.sin(Math.min(1, elapsed * 2.5) * Math.PI) * 0.7;
        arms[1].rotation.x = -0.4;
        head.rotation.z = 0.08;
        break;
      case 'bonk':
        head.rotation.z = oscillation * 0.12 * Math.exp(-elapsed * 2.5);
        head.rotation.x = 0.13 * Math.exp(-elapsed * 3);
        pose.scale.y = 1 - Math.max(0, Math.sin(elapsed * 12)) * 0.06 * Math.exp(-elapsed * 3);
        break;
      case 'wobble':
        pose.rotation.z = oscillation * 0.06;
        head.rotation.z = -oscillation * 0.13;
        break;
      case 'laugh':
        head.rotation.x = -0.09 + Math.sin(elapsed * 18) * 0.05;
        torso.scale.y = ANATOMY.torso[1] * (1 + Math.sin(elapsed * 18) * 0.015);
        ears.forEach((ear, index) => { ear.rotation.z += Math.sin(elapsed * 21 + index) * 0.08; });
        break;
      case 'windup': head.rotation.x = -Math.min(0.2, elapsed * 0.45); break;
      case 'sneeze':
        head.rotation.x = Math.max(0, oscillation) * 0.3;
        torso.rotation.x = Math.max(0, oscillation) * 0.08;
        break;
      case 'wind':
        head.rotation.z = -0.13 + Math.sin(elapsed * 20) * 0.035;
        fringe.rotation.z = -0.10 + Math.sin(elapsed * 30) * 0.065;
        tail.rotation.z = -0.14;
        break;
      case 'hiccup':
        head.rotation.x = -Math.max(0, oscillation) * 0.13;
        pose.scale.y = 1 + Math.max(0, oscillation) * 0.025;
        break;
      case 'jump':
        pose.position.y = Math.abs(Math.sin(Math.min(1, elapsed / 0.55) * Math.PI)) * 0.23;
        arms[-1].rotation.z = 0.2;
        arms[1].rotation.z = -0.2;
        break;
      case 'dance':
        pose.rotation.z = Math.sin(time * 7) * 0.055;
        arms[-1].rotation.z = 0.15 + Math.sin(time * 7) * 0.15;
        arms[1].rotation.z = -0.15 + Math.sin(time * 7) * 0.15;
        break;
      default: break;
    }
  }
  update();
  return { group, update, anchors, dispose() {
    if (disposed) return;
    disposed = true;
    geometries.forEach(geometry => geometry.dispose());
    kit.dispose();
    group.removeFromParent();
  } };
}
