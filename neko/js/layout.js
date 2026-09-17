const SCENE = Object.freeze({ width: 1600, height: 1000 });
const CHARACTER = Object.freeze({ width: 240, height: 420 });
const GROUND_Y = 786;
// Reserve both places even when no friend is visiting. Bounds describe the
// painted silhouettes; the SVG canvases include unused transparent margins.
const PLACES = Object.freeze({
  host: { x: 770, height: 530, left: 36, right: 239, footY: 370 },
  guest: { x: 1070, height: 480, left: 43, right: 238, footY: 395 },
});

function sceneViewport(stage) {
  const { width, height } = stage.getBoundingClientRect();
  const scale = Math.max(width / SCENE.width, height / SCENE.height);
  return {
    width,
    height,
    scale,
    offsetX: (width - SCENE.width * scale) / 2,
    offsetY: (height - SCENE.height * scale) / 2,
  };
}

// Match the room SVG's xMidYMid slice transform so the actor and room share
// one coordinate system, including when a narrow viewport crops the room.
export function getScenePoint(stage, x, y) {
  const { scale, offsetX, offsetY } = sceneViewport(stage);
  return { x: offsetX + x * scale, y: offsetY + y * scale, scale };
}

export function placeCast(stage, actors) {
  const viewport = sceneViewport(stage);
  const placements = new Map();
  if (!viewport.width || !viewport.height) return placements;
  const menuClearance = viewport.width <= 600 ? 78 : 104;
  const availableWidth = Math.max(80, viewport.width - menuClearance - 18);
  const hostLeft = PLACES.host.x + (PLACES.host.left - CHARACTER.width / 2) * PLACES.host.height / CHARACTER.height;
  const guestRight = PLACES.guest.x + (PLACES.guest.right - CHARACTER.width / 2) * PLACES.guest.height / CHARACTER.height;
  const groupWidth = guestRight - hostLeft;
  const groupCenter = (hostLeft + guestRight) / 2;
  const castScale = Math.min(
    viewport.scale,
    viewport.height * 0.62 / PLACES.host.height,
    availableWidth / groupWidth,
  );
  const anchor = getScenePoint(stage, groupCenter, GROUND_Y);
  const halfGroup = groupWidth * castScale / 2;
  const centerX = Math.max(
    menuClearance + halfGroup,
    Math.min(anchor.x, viewport.width - halfGroup - 18),
  );
  const groundY = Math.min(anchor.y, viewport.height - Math.max(24, viewport.height * 0.04));

  for (const [id, actor] of actors) {
    const sitting = id === 'yani';
    const place = sitting ? PLACES.host : PLACES.guest;
    const height = place.height * castScale;
    const width = height * CHARACTER.width / CHARACTER.height;
    const actorScale = height / CHARACTER.height;
    const groundX = centerX + (place.x - groupCenter) * castScale;

    actor.dataset.pose = sitting ? 'sit' : 'stand';
    actor.style.left = `${groundX}px`;
    actor.style.top = `${groundY - place.footY * actorScale}px`;
    actor.style.width = `${width}px`;
    actor.style.height = `${height}px`;
    actor.style.bottom = 'auto';
    actor.style.transform = 'translateX(-50%)';
    actor.style.setProperty('--foot-y', `${place.footY / CHARACTER.height * 100}%`);
    placements.set(id, { groundX, groundY, scale: viewport.scale, actorScale });
  }

  return placements;
}
