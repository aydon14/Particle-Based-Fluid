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
  // keep layout parity with compute:
  tensileK : f32,
  tensileN : f32,
  tensileDeltaQ : f32,
  centerGravity : f32,
  // add: color mode
  colorMode : f32,
};

@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var<storage, read> Particles : array<Particle>;

struct VSOut {
  @builtin(position) pos : vec4f,
  @location(0) speed : f32,
  @location(1) local : vec2f,
  @location(2) dens : f32,
};

@vertex
fn vsMain(@location(0) vert : vec2f, @builtin(instance_index) inst : u32) -> VSOut {
  let p = Particles[inst].pos;
  let v = Particles[inst].vel;

  let center = vec2f(p.x / U.halfW, p.y / U.halfH);
  let rx = U.particleRadius / U.halfW;
  let ry = U.particleRadius / U.halfH;

  var out : VSOut;
  out.pos = vec4f(center + vec2f(vert.x * rx, vert.y * ry), 0.0, 1.0);
  out.speed = length(v);
  out.local = vert;
  out.dens = Particles[inst].dens;
  return out;
}

fn paletteSpeed(speed : f32) -> vec3f {
  let maxv = max(U.velocityColorMax, 1.0);
  let t = clamp(speed / maxv, 0.0, 1.0);
  if (t < 0.25) {
    let k = t / 0.25;
    return mix(vec3f(34.0,87.0,185.0), vec3f(76.0,255.0,144.0), k) / 255.0;
  } else if (t < 0.75) {
    let k = (t - 0.25) / 0.5;
    return mix(vec3f(76.0,255.0,144.0), vec3f(255.0,237.0,0.0), k) / 255.0;
  } else {
    let k = (t - 0.75) / 0.25;
    return mix(vec3f(255.0,237.0,0.0), vec3f(247.0,73.0,8.0), k) / 255.0;
  }
}

fn paletteDensity(dens : f32) -> vec3f {
  // map relative density to colors
  let td = max(U.targetDensity, 0.01);
  let t = clamp((dens - td) / td, 0.0, 1.0);
  // cool-to-hot
  if (t < 0.5) {
    let k = t / 0.5;
    return mix(vec3f(40.0,120.0,255.0), vec3f(0.0,255.0,200.0), k) / 255.0;
  } else {
    let k = (t - 0.5) / 0.5;
    return mix(vec3f(0.0,255.0,200.0), vec3f(255.0,80.0,0.0), k) / 255.0;
  }
}

@fragment
fn fsMain(@location(0) speed : f32, @location(1) local : vec2f, @location(2) dens : f32) -> @location(0) vec4f {
  let r2 = dot(local, local);
  if (r2 > 1.0) { discard; }
  let col = select(paletteSpeed(speed), paletteDensity(dens), U.colorMode >= 0.5);
  return vec4f(col, 1.0);
}