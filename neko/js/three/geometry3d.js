import * as THREE from '../../assets/vendor/three/three.module.min.js';

const TAU = Math.PI * 2;

function makeGeometry(positions, uvs, indices) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

// Smooth interpolation through measured cross sections, preserving each endpoint.
function sectionAt(profile, y) {
  let index = 0;
  while (index < profile.length - 2 && profile[index + 1][0] < y) index++;
  const left = profile[index], right = profile[index + 1];
  const t = THREE.MathUtils.clamp((y - left[0]) / (right[0] - left[0]), 0, 1);
  const before = profile[Math.max(0, index - 1)];
  const after = profile[Math.min(profile.length - 1, index + 2)];
  const result = [y];
  for (let axis = 1; axis < 3; axis++) {
    const d = right[0] - left[0];
    const m0 = (right[axis] - before[axis]) / (right[0] - before[0]) * d;
    const m1 = (after[axis] - left[axis]) / (after[0] - left[0]) * d;
    result.push(Math.max(0, (2 * t ** 3 - 3 * t ** 2 + 1) * left[axis] +
      (t ** 3 - 2 * t ** 2 + t) * m0 + (-2 * t ** 3 + 3 * t ** 2) * right[axis] +
      (t ** 3 - t ** 2) * m1));
  }
  return result;
}

/** A single continuous, UV-unwrapped cross-section surface. */
export function loft(profile, { radial = 48, rows = 40, sculpt } = {}) {
  const positions = [], uvs = [], indices = [];
  const low = profile[0][0], high = profile.at(-1)[0];
  for (let row = 0; row <= rows; row++) {
    const v = row / rows, y = low + (high - low) * v;
    const [, width, depth] = sectionAt(profile, y);
    for (let column = 0; column <= radial; column++) {
      const u = column / radial, phi = (u - 0.5) * TAU;
      let point = [Math.sin(phi) * width, y, Math.cos(phi) * depth];
      if (sculpt) point = sculpt(point, phi, v);
      positions.push(...point);
      uvs.push(u, v);
      if (row < rows && column < radial) {
        const current = row * (radial + 1) + column;
        indices.push(current, current + 1, current + radial + 1,
          current + 1, current + radial + 2, current + radial + 1);
      }
    }
  }
  return makeGeometry(positions, uvs, indices);
}

/** Curved anatomical limbs, cloth sleeves, tails, and sharply tapered hair. */
export function sweep(points, widths, { depths = widths, radial = 12, rows = 24, ribbon = false, folds = 0, capped = true } = {}) {
  const path = new THREE.CatmullRomCurve3(points.map(point => new THREE.Vector3(...point)));
  const positions = [], uvs = [], indices = [];
  const interpolate = (values, t) => {
    const at = Math.min(values.length - 1.000001, t * (values.length - 1));
    const low = Math.floor(at), mix = at - low;
    return THREE.MathUtils.lerp(values[low], values[low + 1], mix);
  };
  for (let row = 0; row <= rows; row++) {
    const t = row / rows;
    const center = path.getPoint(t), tangent = path.getTangent(t).normalize();
    const reference = Math.abs(tangent.z) > 0.94 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
    const horizontal = new THREE.Vector3().crossVectors(tangent, reference).normalize();
    const vertical = new THREE.Vector3().crossVectors(tangent, horizontal).normalize();
    const width = interpolate(widths, t), depth = interpolate(depths, t);
    for (let column = 0; column <= radial; column++) {
      const u = column / radial, angle = u * TAU;
      let dx = Math.cos(angle), dz = Math.sin(angle);
      if (ribbon) {
        // A ridged lens cross section, with a fine blade edge rather than a tube.
        dz = Math.sign(dz) * Math.pow(Math.abs(dz), 0.65);
        dz *= dz > 0 ? 1 : 0.55;
      }
      const wrinkle = 1 + Math.sin(angle * 4 + t * 22) * folds * Math.sin(Math.PI * t);
      const point = center.clone().addScaledVector(horizontal, dx * width * wrinkle)
        .addScaledVector(vertical, dz * depth * wrinkle);
      positions.push(point.x, point.y, point.z);
      uvs.push(u, t);
      if (row < rows && column < radial) {
        const current = row * (radial + 1) + column;
        indices.push(current, current + 1, current + radial + 1,
          current + 1, current + radial + 2, current + radial + 1);
      }
    }
  }
  if (capped) {
    // Separate rim vertices retain the side normals and make a sealed end face.
    // Garment shoulders are buried in the torso, but still need solid end caps
    // when an animated arm rotates out from underneath the shoulder seam.
    for (const end of [0, 1]) {
      const center = path.getPoint(end);
      const centerIndex = positions.length / 3;
      positions.push(center.x, center.y, center.z);
      uvs.push(0.5, 0.5);
      const base = end * rows * (radial + 1);
      for (let column = 0; column <= radial; column++) {
        const source = (base + column) * 3;
        positions.push(positions[source], positions[source + 1], positions[source + 2]);
        const angle = column / radial * TAU;
        uvs.push(0.5 + Math.cos(angle) * 0.5, 0.5 + Math.sin(angle) * 0.5);
        if (column < radial) {
          if (end) indices.push(centerIndex, centerIndex + column + 1, centerIndex + column + 2);
          else indices.push(centerIndex, centerIndex + column + 2, centerIndex + column + 1);
        }
      }
    }
  }
  return makeGeometry(positions, uvs, indices);
}

export function combine(geometries) {
  const positions = [], uvs = [], normals = [], indices = [];
  let offset = 0;
  for (const geo of geometries) {
    positions.push(...geo.attributes.position.array);
    uvs.push(...geo.attributes.uv.array);
    normals.push(...geo.attributes.normal.array);
    if (geo.index) {
      for (const index of geo.index.array) indices.push(index + offset);
    } else {
      for (let index = 0; index < geo.attributes.position.count; index++) indices.push(index + offset);
    }
    offset += geo.attributes.position.count;
    geo.dispose();
  }
  const geometry = makeGeometry(positions, uvs, indices);
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  return geometry;
}

export function tailoredPanel(points, depth = 0.04, bevel = 0.012) {
  const shape = new THREE.Shape();
  points.forEach(([x, y], index) => index ? shape.lineTo(x, y) : shape.moveTo(x, y));
  shape.closePath();
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel,
    bevelSegments: 3, curveSegments: 8, steps: 1,
  });
  geometry.computeBoundingBox();
  const { min, max } = geometry.boundingBox;
  const uv = geometry.attributes.uv, position = geometry.attributes.position;
  for (let index = 0; index < position.count; index++) {
    uv.setXY(index, (position.getX(index) - min.x) / (max.x - min.x),
      (position.getY(index) - min.y) / (max.y - min.y));
  }
  return geometry;
}

export const HEAD_PROFILE = [
  [-0.62, 0, 0], [-0.58, 0.18, 0.19], [-0.49, 0.36, 0.32],
  [-0.36, 0.515, 0.44], [-0.18, 0.61, 0.505], [0, 0.665, 0.535],
  [0.2, 0.645, 0.515], [0.38, 0.56, 0.45], [0.55, 0.37, 0.30], [0.65, 0, 0],
];

export function sculptedHead() {
  return loft(HEAD_PROFILE, { radial: 96, rows: 72, sculpt: ([x, y, z], phi) => {
    const front = Math.max(0, Math.cos(phi));
    // The cheek, brow and nose forms belong to the face topology itself.
    const nose = Math.exp(-((phi / 0.12) ** 2 + ((y + 0.165) / 0.085) ** 2));
    const bridge = Math.exp(-((phi / 0.07) ** 2 + ((y + 0.04) / 0.14) ** 2));
    const cheeks = Math.exp(-(((Math.abs(phi) - 0.48) / 0.18) ** 2)) * Math.exp(-(((y + 0.19) / 0.12) ** 2));
    z += front * (nose * 0.055 + bridge * 0.018 + cheeks * 0.011);
    return [x, y, z];
  } });
}

/** Rounded ear cartilage with a lining that follows its curved front surface. */
export function sculptedEar() {
  const profile = [[0, 0.18, 0.085], [0.12, 0.17, 0.105],
    [0.26, 0.13, 0.08], [0.40, 0.075, 0.046], [0.55, 0, 0]];
  const shell = loft(profile, { radial: 40, rows: 32,
    sculpt: ([x, y, z]) => [x - 0.033 * y, y, z - 0.06 * y] });
  const positions = [], uvs = [], indices = [], rows = 22, columns = 16;
  for (let row = 0; row <= rows; row++) {
    const v = row / rows, y = 0.07 + v * 0.405;
    const [, width, depth] = sectionAt(profile, y);
    const inset = Math.sin(Math.PI * v) ** 0.42 * 0.72;
    for (let column = 0; column <= columns; column++) {
      const u = column / columns, across = (u * 2 - 1) * inset;
      positions.push(across * width - 0.033 * y, y,
        Math.sqrt(1 - across * across) * depth - 0.06 * y + 0.004);
      uvs.push(u, v);
      if (row < rows && column < columns) {
        const i = row * (columns + 1) + column;
        indices.push(i, i + 1, i + columns + 1, i + 1, i + columns + 2, i + columns + 1);
      }
    }
  }
  return { shell, lining: makeGeometry(positions, uvs, indices) };
}

export function scalp() {
  const radial = 64, rows = 36, positions = [], uvs = [], indices = [];
  for (let row = 0; row <= rows; row++) {
    const t = row / rows;
    for (let column = 0; column <= radial; column++) {
      const u = column / radial, phi = (u - 0.5) * TAU;
      const cut = -0.40 + 0.65 * Math.max(0, Math.cos(phi)) ** 1.3;
      const y = THREE.MathUtils.lerp(cut, 0.70, t);
      const [, width, depth] = sectionAt(HEAD_PROFILE, Math.min(0.649, y - 0.04));
      const cap = Math.sin(Math.PI * 0.5 * (1 - t));
      positions.push(Math.sin(phi) * (width + 0.055 * cap), y,
        Math.cos(phi) * (depth + 0.055 * cap) - 0.025);
      uvs.push(u, t);
      if (row < rows && column < radial) {
        const current = row * (radial + 1) + column;
        indices.push(current, current + 1, current + radial + 1,
          current + 1, current + radial + 2, current + radial + 1);
      }
    }
  }
  return makeGeometry(positions, uvs, indices);
}
