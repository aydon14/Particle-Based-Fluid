struct Particle {
  pos : vec2f,
  vel : vec2f,
  pred : vec2f,
  dens : f32,
  nearDens : f32,
};

struct Uniforms {
  time : f32, dt : f32, particleRadius : f32, pad0 : f32,
  halfW : f32, halfH : f32,
  boundsW : f32, boundsH : f32,
  gridW : f32, gridH : f32,
  smoothingRadius : f32, targetDensity : f32, pressureMultiplier : f32, nearPressureMultiplier : f32,
  viscosityStrength : f32, collisionDamping : f32, gravity : f32, interactionRadius : f32,
  interactionStrength : f32, mouseX : f32, mouseY : f32, pixelsPerUnit : f32,
  velocityColorMax : f32, iterationsPerFrame : f32, numParticles : f32,
  tensileK : f32,
  tensileN : f32,
  tensileDeltaQ : f32,
  centerGravity : f32,
  colorMode : f32,
};

@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var<storage, read_write> Particles : array<Particle>;
@group(0) @binding(2) var<storage, read_write> GridCounts : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> GridIndices : array<u32>;

const WORKGROUP_SIZE : u32 = 256u;
const MAX_PER_CELL : u32 = 256u;   // must match JS
const PAD_CELLS : i32 = 2;
const CELL_SCALE : f32 = 0.5;      // smaller grid cells to reduce contention

// Kernels
fn kernelSpiky2(v : f32, r : f32) -> f32 {
  return v * v * (6.0 / (3.1415926 * r * r * r * r));
}
fn kernelSpiky3(v : f32, r : f32) -> f32 {
  return v * v * v * (10.0 / (3.1415926 * r * r * r * r * r));
}
fn kernelPoly6(v2 : f32, r : f32) -> f32 {
  return (v2 * v2 * v2) * (4.0 / (3.1415926 * pow(r, 8.0)));
}

fn inRange(ix : i32, iy : i32, w : i32, h : i32) -> bool {
  return ix >= 0 && iy >= 0 && ix < w && iy < h;
}

fn worldToCellPadded(p : vec2f) -> vec2<i32> {
  let cellSize = U.smoothingRadius * CELL_SCALE;
  let origin = vec2f(-U.halfW, -U.halfH) - vec2f(f32(PAD_CELLS), f32(PAD_CELLS)) * cellSize;
  let rel = p - origin;
  let ix = i32(floor(rel.x / cellSize));
  let iy = i32(floor(rel.y / cellSize));
  return vec2<i32>(ix, iy);
}

fn cellIndex(ix : i32, iy : i32) -> u32 {
  return u32(iy) * u32(U.gridW) + u32(ix);
}

// ceil(R / (R*CELL_SCALE)) => ceil(1/CELL_SCALE)
fn ringForRadius(cellScale : f32) -> i32 {
  return i32(ceil(1.0 / cellScale)); // 2 for 0.5 -> 5x5 neighborhood
}

// 1) Predict: gravity (downward or radial toward center), mouse handled separately in applyMouse
@compute @workgroup_size(256)
fn predict(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u32(U.numParticles)) { return; }

  var p = Particles[i].pos;
  var v = Particles[i].vel;

  let dt = U.dt;

  // Gravity mode
  if (U.centerGravity > 0.5) {
    // Radial gravity toward origin with softening to avoid singularity at center.
    // Magnitude uses |gravity|; switch to inv2 for 1/r^2 behavior if desired.
    let d = -p;
    let dist = length(d);
    let eps = U.smoothingRadius * 0.5;
    let inv = 1.0 / max(dist, eps);   // 1/r falloff. For 1/r^2: let inv2 = 1.0 / max(dist*dist, eps*eps);
    let gmag = abs(U.gravity);
    let gvec = d * (gmag * inv);
    v = v + gvec * dt;
  } else {
    v = v + vec2f(0.0, U.gravity) * dt;
  }

  let pred = p + v * dt;

  Particles[i].vel = v;
  Particles[i].pred = pred;
}

// 2) Clear grid
@compute @workgroup_size(256)
fn clearGrid(@builtin(global_invocation_id) gid : vec3<u32>) {
  let numCells = u32(U.gridW * U.gridH);
  let idx = gid.x;
  if (idx >= numCells) { return; }
  atomicStore(&GridCounts[idx], 0u);
}

// 3) Build grid from predicted; cyclic write to avoid bias on overflow
@compute @workgroup_size(256)
fn buildGrid(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u32(U.numParticles)) { return; }
  let c = worldToCellPadded(Particles[i].pred);
  if (!inRange(c.x, c.y, i32(U.gridW), i32(U.gridH))) { return; }
  let ci = cellIndex(c.x, c.y);
  let slot = atomicAdd(&GridCounts[ci], 1u);
  let wslot = slot % MAX_PER_CELL;
  GridIndices[ci * MAX_PER_CELL + wslot] = i;
}

// 4) Mouse-only pass: affect only particles in cells near the mouse
@compute @workgroup_size(64)
fn applyMouse(
  @builtin(workgroup_id) wg_id : vec3<u32>,
  @builtin(local_invocation_id) lid : vec3<u32>
) {
  if (abs(U.interactionStrength) <= 0.0) { return; }

  let cellSize = U.smoothingRadius * CELL_SCALE;
  let ring = i32(ceil(U.interactionRadius / cellSize));
  let side = 2 * ring + 1;
  let total = side * side;
  let cidx = i32(wg_id.x);
  if (cidx >= total) { return; }

  let ox = (cidx % side) - ring;
  let oy = (cidx / side) - ring;

  let mouseCell = worldToCellPadded(vec2f(U.mouseX, U.mouseY));
  let cx = mouseCell.x + ox;
  let cy = mouseCell.y + oy;
  if (!inRange(cx, cy, i32(U.gridW), i32(U.gridH))) { return; }

  let ci = cellIndex(cx, cy);
  let count = min(atomicLoad(&GridCounts[ci]), MAX_PER_CELL);

  let tid = u32(lid.x);
  let step = u32(64);

  let r = U.interactionRadius;
  let r2 = r * r;
  let dt = U.dt;

  for (var k = tid; k < count; k = k + step) {
    let j = GridIndices[ci * MAX_PER_CELL + k];

    var vj = Particles[j].vel;
    let pj = Particles[j].pred;

    let d = vec2f(U.mouseX, U.mouseY) - pj;
    let sqrDst = dot(d, d);
    if (sqrDst < r2 && sqrDst > 0.0) {
      let dst = sqrt(sqrDst);
      let invDst = 1.0 / dst;
      let edgeT = dst / r;
      let centreT = 1.0 - edgeT;

      // Mouse impulse only (predict already did gravity)
      vj = vj + (d * invDst * centreT * U.interactionStrength - vj * centreT) * dt;
      Particles[j].vel = vj;
    }
  }
}

// 5) Densities from predicted positions
@compute @workgroup_size(256)
fn computeDensities(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u32(U.numParticles)) { return; }

  let radius = U.smoothingRadius;
  let radiusSq = radius * radius;
  var dens = 0.0;
  var nearDens = 0.0;

  let c = worldToCellPadded(Particles[i].pred);
  let ring = ringForRadius(CELL_SCALE);

  for (var dy = -ring; dy <= ring; dy = dy + 1) {
    for (var dx = -ring; dx <= ring; dx = dx + 1) {
      let ix = c.x + dx;
      let iy = c.y + dy;
      if (!inRange(ix, iy, i32(U.gridW), i32(U.gridH))) { continue; }
      let ci = cellIndex(ix, iy);
      let count = min(atomicLoad(&GridCounts[ci]), MAX_PER_CELL);
      for (var k : u32 = 0u; k < count; k = k + 1u) {
        let j = GridIndices[ci * MAX_PER_CELL + k];
        let d = Particles[j].pred - Particles[i].pred;
        let sqrDst = dot(d, d);
        if (sqrDst < radiusSq) {
          let dst = sqrt(sqrDst);
          let v = radius - dst;
          dens += kernelSpiky2(v, radius);
          nearDens += kernelSpiky3(v, radius);
        }
      }
    }
  }

  Particles[i].dens = dens;
  Particles[i].nearDens = nearDens;
}

// 6) Pressure/viscosity + tensile correction, integrate and bounce
@compute @workgroup_size(256)
fn updateVelPos(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= u32(U.numParticles)) { return; }

  let radius = U.smoothingRadius;
  let radiusSq = radius * radius;
  let targetDens = max(U.targetDensity, 0.01);
  let pressMult = U.pressureMultiplier;
  let nearPressMult = U.nearPressureMultiplier;
  let viscStr = U.viscosityStrength;
  let damp = -U.collisionDamping;

  var p = Particles[i].pos;
  var v = Particles[i].vel;
  let pred = Particles[i].pred;

  let dens = max(Particles[i].dens, 0.01);
  let nearDens = max(Particles[i].nearDens, 0.01);
  let press = (dens - targetDens) * pressMult;
  let nearPress = nearDens * nearPressMult;

  var fx = 0.0;
  var fy = 0.0;
  var viscX = 0.0;
  var viscY = 0.0;

  // Tensile correction precompute
  let dq = clamp(U.tensileDeltaQ, 0.05, 0.6) * radius;
  let v2q = max(radiusSq - dq * dq, 0.0);
  let Wq = max(kernelPoly6(v2q, radius), 1e-8);

  let c = worldToCellPadded(pred);
  let ring = ringForRadius(CELL_SCALE);

  for (var dy = -ring; dy <= ring; dy = dy + 1) {
    for (var dx = -ring; dx <= ring; dx = dx + 1) {
      let ix = c.x + dx;
      let iy = c.y + dy;
      if (!inRange(ix, iy, i32(U.gridW), i32(U.gridH))) { continue; }
      let ci = cellIndex(ix, iy);
      let count = min(atomicLoad(&GridCounts[ci]), MAX_PER_CELL);
      for (var k : u32 = 0u; k < count; k = k + 1u) {
        let j = GridIndices[ci * MAX_PER_CELL + k];
        if (j == i) { continue; }
        let d = Particles[j].pred - pred;
        let sqrDst = dot(d, d);
        if (sqrDst < radiusSq && sqrDst > 0.0) {
          let dst = sqrt(sqrDst);
          let invDst = 1.0 / dst;

          let nDens = max(Particles[j].dens, 0.01);
          let nNearDens = max(Particles[j].nearDens, 0.01);
          let nPress = (nDens - targetDens) * pressMult;
          let nNearPress = nNearDens * nearPressMult;

          let sharedPress = 0.5 * (press + nPress);
          let sharedNearPress = 0.5 * (nearPress + nNearPress);

          let vspiky = radius - dst;
          let grad = -vspiky * (12.0 / (pow(radius, 4.0) * 3.1415926));
          let nearGrad = -(vspiky * vspiky) * (30.0 / (pow(radius, 5.0) * 3.1415926));

          let fpair = (grad * sharedPress / nDens + nearGrad * sharedNearPress / nNearDens) * invDst;

          fx += d.x * fpair;
          fy += d.y * fpair;

          // Viscosity + tensile correction
          let vdiff = Particles[j].vel - v;
          let v2 = radiusSq - dst * dst;
          if (v2 > 0.0) {
            let influence = kernelPoly6(v2, radius);
            viscX += vdiff.x * influence;
            viscY += vdiff.y * influence;

            let W = influence;
            let s_corr = -U.tensileK * pow(W / Wq, U.tensileN);
            fx += d.x * invDst * s_corr;
            fy += d.y * invDst * s_corr;
          }
        }
      }
    }
  }

  let dt = U.dt;
  let accel = 1.0 / dens;
  v = v + vec2f(fx, fy) * accel * dt;
  v = v + vec2f(viscX, viscY) * viscStr * dt;

  // Integrate and bounce
  p = p + v * dt;
  if (abs(p.x) >= U.halfW) { p.x = U.halfW * sign(p.x); v.x = v.x * damp; }
  if (abs(p.y) >= U.halfH) { p.y = U.halfH * sign(p.y); v.y = v.y * damp; }

  Particles[i].pos = p;
  Particles[i].vel = v;
}