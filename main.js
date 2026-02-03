async function loadText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  return res.text();
}

const CONFIG = {
  numParticles: 5000,
  gravity: -12,
  timeScale: 1,
  iterationsPerFrame: 2,      // base substeps; we add dynamic substeps when size shrinks
  maxTimestepFPS: 60,
  smoothingRadius: 0.35,      // physics neighbor radius (will change with size)
  targetDensity: 55,
  pressureMultiplier: 500,
  nearPressureMultiplier: 5,
  viscosityStrength: 0.03,
  collisionDamping: 0.95,
  boundsWidth: 16,
  boundsHeight: 9,
  spawnCenter: { x: 0, y: 0.66 },
  spawnSize: { x: 6.42, y: 4.39 },
  jitterStrength: 0.03,
  interactionRadius: 2,
  interactionStrength: 90,
  particleRadius: 0.06,       // visual radius; linked to physics
  velocityColorMax: 3.5,
  pixelsPerUnit: 120,
  useGradient: false,
  maxPerCell: 256,            // MUST match shader
  tensileK: 0.005,       
  tensileN: 4.0,          
  tensileDeltaQ: 0.27,
  centerGravity: false,
};

const CELL_SCALE = 0.5;

// Baselines for scaling
const BASES = {
  particleRadius: CONFIG.particleRadius,
  smoothingRadius: CONFIG.smoothingRadius,
  targetDensity: CONFIG.targetDensity,
  pressureMultiplier: CONFIG.pressureMultiplier,
  nearPressureMultiplier: CONFIG.nearPressureMultiplier,
  viscosityStrength: CONFIG.viscosityStrength,
  ratioSRtoPR: CONFIG.smoothingRadius / CONFIG.particleRadius, // ~5.833...
};

const canvas = document.getElementById("canvas");

async function initWebGPU() {
  const [computeWGSL, renderWGSL] = await Promise.all([
    loadText("./shaders/compute.wgsl"),
    loadText("./shaders/render.wgsl"),
  ]);

  if (!navigator.gpu) throw new Error("WebGPU not supported");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No GPU adapter found");
  const device = await adapter.requestDevice();

  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();

  const scale = CONFIG.pixelsPerUnit;
  canvas.width = Math.floor(CONFIG.boundsWidth * scale);
  canvas.height = Math.floor(CONFIG.boundsHeight * scale);

  context.configure({ device, format, alphaMode: "premultiplied" });

  const halfW = CONFIG.boundsWidth * 0.5;
  const halfH = CONFIG.boundsHeight * 0.5;

  // Grid parameters (mutable when smoothingRadius changes)
  const GRID_PAD_CELLS = 2;
  let gridW = 0, gridH = 0, numCells = 0;
  let gridCountsBuffer, gridIndicesBuffer;

  function recomputeGridDimensions() {
    const cellSize = CONFIG.smoothingRadius * CELL_SCALE;
    gridW = Math.max(1, Math.ceil(CONFIG.boundsWidth / cellSize) + GRID_PAD_CELLS * 2);
    gridH = Math.max(1, Math.ceil(CONFIG.boundsHeight / cellSize) + GRID_PAD_CELLS * 2);
    numCells = gridW * gridH;
  }

  // Buffers
  const particleStrideFloats = 2 + 2 + 2 + 1 + 1; // pos(2), vel(2), pred(2), dens, near
  const particleStrideBytes = particleStrideFloats * 4;

  let particlesBuffer = device.createBuffer({
    size: CONFIG.numParticles * particleStrideBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
  });

  const uniformBuffer = device.createBuffer({
    size: 512,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  function createGridBuffers() {
    recomputeGridDimensions();
    gridCountsBuffer?.destroy?.();
    gridIndicesBuffer?.destroy?.();
    gridCountsBuffer = device.createBuffer({
      size: numCells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    gridIndicesBuffer = device.createBuffer({
      size: numCells * CONFIG.maxPerCell * 4,
      usage: GPUBufferUsage.STORAGE,
    });
  }

  createGridBuffers(); // initial

  const quadVertices = new Float32Array([ -1, -1,  1, -1,  1,  1,  -1, -1,  1,  1, -1,  1 ]);
  const quadVB = device.createBuffer({
    size: quadVertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(quadVB, 0, quadVertices);

  const computeModule = device.createShaderModule({ code: computeWGSL });
  const renderModule = device.createShaderModule({ code: renderWGSL });

  const computeBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]
  });

  let computeBindGroup = device.createBindGroup({
    layout: computeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particlesBuffer } },
      { binding: 2, resource: { buffer: gridCountsBuffer } },
      { binding: 3, resource: { buffer: gridIndicesBuffer } },
    ]
  });

  const predictPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
    compute: { module: computeModule, entryPoint: "predict" }
  });
  const clearPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
    compute: { module: computeModule, entryPoint: "clearGrid" }
  });
  const buildGridPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
    compute: { module: computeModule, entryPoint: "buildGrid" }
  });
  const applyMousePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
    compute: { module: computeModule, entryPoint: "applyMouse" }
  });
  const densityPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
    compute: { module: computeModule, entryPoint: "computeDensities" }
  });
  const updatePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
    compute: { module: computeModule, entryPoint: "updateVelPos" }
  });

  const renderBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
    ]
  });

  let renderBindGroup = device.createBindGroup({
    layout: renderBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: particlesBuffer } },
    ]
  });

  const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [renderBindGroupLayout] }),
    vertex: { module: renderModule, entryPoint: "vsMain",
      buffers: [{ arrayStride: 2 * 4, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: renderModule, entryPoint: "fsMain",
      targets: [{ format,
        blend: {
          color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
          alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
        } } ] },
    primitive: { topology: "triangle-list" }
  });

  function spawnData(count) {
    const ratio = CONFIG.spawnSize.x / (CONFIG.spawnSize.x + CONFIG.spawnSize.y);
    const m = Math.sqrt(count / (ratio * (1 - ratio)));
    const nx = Math.ceil(ratio * m);
    const ny = Math.ceil((1 - ratio) * m);

    const data = new Float32Array(count * particleStrideFloats);
    let idx = 0;
    for (let y = 0; y < ny && idx < count; y++) {
      for (let x = 0; x < nx && idx < count; x++) {
        const fx = nx > 1 ? x / (nx - 1) : 0.5;
        const fy = ny > 1 ? y / (ny - 1) : 0.5;
        const px = (fx - 0.5) * CONFIG.spawnSize.x + CONFIG.spawnCenter.x;
        const py = (fy - 0.5) * CONFIG.spawnSize.y + CONFIG.spawnCenter.y;
        const angle = Math.random() * Math.PI * 2;
        const jitter = (Math.random() - 0.5) * CONFIG.jitterStrength;
        const ox = Math.cos(angle) * jitter;
        const oy = Math.sin(angle) * jitter;

        const base = idx * particleStrideFloats;
        data[base + 0] = px + ox;
        data[base + 1] = py + oy;
        data[base + 2] = 0;
        data[base + 3] = 0;
        data[base + 4] = px + ox;
        data[base + 5] = py + oy;
        data[base + 6] = 0;
        data[base + 7] = 0;
        idx++;
      }
    }
    return data;
  }

  function writeUniforms(dt) {
    const u = new Float32Array(64);
    let i = 0;
    const maxDt = 1 / CONFIG.maxTimestepFPS;

    u[i++] = performance.now() * 0.001;
    u[i++] = Math.min(dt * CONFIG.timeScale, maxDt);
    u[i++] = CONFIG.particleRadius; // visual radius
    u[i++] = 0.0;

    u[i++] = halfW;
    u[i++] = halfH;

    u[i++] = CONFIG.boundsWidth;
    u[i++] = CONFIG.boundsHeight;

    u[i++] = gridW;
    u[i++] = gridH;

    u[i++] = CONFIG.smoothingRadius;        // physics radius
    u[i++] = CONFIG.targetDensity;
    u[i++] = CONFIG.pressureMultiplier;
    u[i++] = CONFIG.nearPressureMultiplier;

    u[i++] = CONFIG.viscosityStrength;
    u[i++] = CONFIG.collisionDamping;
    u[i++] = CONFIG.gravity;
    u[i++] = CONFIG.interactionRadius;

    const strength = mouseLeft ? CONFIG.interactionStrength : (mouseRight ? -CONFIG.interactionStrength : 0);
    u[i++] = strength;

    u[i++] = mouseX;
    u[i++] = mouseY;

    u[i++] = CONFIG.pixelsPerUnit;
    u[i++] = CONFIG.velocityColorMax;

    u[i++] = CONFIG.iterationsPerFrame; // not used by shader but kept for layout stability
    u[i++] = CONFIG.numParticles;

    u[i++] = CONFIG.tensileK;
    u[i++] = CONFIG.tensileN;
    u[i++] = CONFIG.tensileDeltaQ;

    u[i++] = CONFIG.centerGravity ? 1.0 : 0.0;

    device.queue.writeBuffer(uniformBuffer, 0, u.buffer);
  }

  // Init data
  device.queue.writeBuffer(particlesBuffer, 0, spawnData(CONFIG.numParticles).buffer);

  async function resizeParticles(newCount) {
    CONFIG.numParticles = newCount;

    const newBuf = device.createBuffer({
      size: newCount * particleStrideBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
    });
    device.queue.writeBuffer(newBuf, 0, spawnData(newCount).buffer);

    particlesBuffer.destroy?.();
    particlesBuffer = newBuf;

    computeBindGroup = device.createBindGroup({
      layout: computeBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: particlesBuffer } },
        { binding: 2, resource: { buffer: gridCountsBuffer } },
        { binding: 3, resource: { buffer: gridIndicesBuffer } },
      ]
    });

    renderBindGroup = device.createBindGroup({
      layout: renderBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: particlesBuffer } },
      ]
    });

    // document.getElementById("particleCount").textContent = CONFIG.numParticles;
  }

  // Always link size to physics with stability scaling
  function applyParticleSize(newPR) {
    CONFIG.particleRadius = newPR;

    // 1) Link physics radius
    CONFIG.smoothingRadius = BASES.ratioSRtoPR * newPR;

    // 2) Stability scaling based on r = smoothingRadius
    const r = CONFIG.smoothingRadius;
    const r0 = BASES.smoothingRadius;
    const k = r / r0;            // < 1 when shrinking

    // Density ~ 1/r^2, so keep targetDensity stable with inverse-square
    CONFIG.targetDensity = BASES.targetDensity * (r0 / r) * (r0 / r);

    // Pressure acceleration tends to grow ~1/r^2, cancel with press multiplier ~ r^2
    CONFIG.pressureMultiplier = BASES.pressureMultiplier * (k * k);

    // Near-pressure behaves stronger as r shrinks; mild linear scale tamps it down
    CONFIG.nearPressureMultiplier = BASES.nearPressureMultiplier * k;

    // Viscosity gets stronger with tiny radii; scale down linearly
    CONFIG.viscosityStrength = BASES.viscosityStrength * k;

    // 3) Rebuild grid for new cell size
    createGridBuffers();
    computeBindGroup = device.createBindGroup({
      layout: computeBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: particlesBuffer } },
        { binding: 2, resource: { buffer: gridCountsBuffer } },
        { binding: 3, resource: { buffer: gridIndicesBuffer } },
      ]
    });

    // 4) Dynamic substeps: more substeps when r is small (keeps dt per substep tiny)
    dynamicSubsteps = Math.max(CONFIG.iterationsPerFrame, Math.min(8, Math.ceil(r0 / r)));
  }

  // Interaction/UI
  let mouseX = 0, mouseY = 0, mouseLeft = false, mouseRight = false;
  let isPaused = false;
  let dynamicSubsteps = CONFIG.iterationsPerFrame;

  const controlsEl = document.getElementById("controls");
  const particleSizeInput = document.getElementById("particleSize");
  const particleSizeVal = document.getElementById("particleSizeVal");

  const centerGravityInput = document.getElementById("centerGravity");
  centerGravityInput.checked = CONFIG.centerGravity;

  particleSizeVal.textContent = CONFIG.particleRadius.toFixed(3);
  particleSizeInput.value = CONFIG.particleRadius;

  centerGravityInput.addEventListener("change", (e) => {
    CONFIG.centerGravity = e.target.checked;
  });

  document.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
      e.preventDefault();
      isPaused = !isPaused;
    }
    if (e.code === "KeyR") {
      device.queue.writeBuffer(particlesBuffer, 0, spawnData(CONFIG.numParticles).buffer);
    }
    if (e.code === "KeyX") {
      controlsEl.classList.toggle("hidden");
    }
  });

  function updateMouse(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const cx = canvas.width * 0.5;
    const cy = canvas.height * 0.5;
    mouseX = (x - cx) / CONFIG.pixelsPerUnit;
    mouseY = -(y - cy) / CONFIG.pixelsPerUnit;
  }
  canvas.addEventListener("mousemove", updateMouse);
  canvas.addEventListener("mousedown", (e) => { e.preventDefault(); if (e.button === 0) mouseLeft = true; if (e.button === 2) mouseRight = true; });
  canvas.addEventListener("mouseup", (e) => { if (e.button === 0) mouseLeft = false; if (e.button === 2) mouseRight = false; });
  canvas.addEventListener("mouseleave", () => { mouseLeft = false; mouseRight = false; });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  document.getElementById("gravity").addEventListener("input", (e) => {
    CONFIG.gravity = parseFloat(e.target.value);
    document.getElementById("gravityVal").textContent = CONFIG.gravity;
  });
  document.getElementById("interaction").addEventListener("input", (e) => {
    CONFIG.interactionStrength = parseFloat(e.target.value);
    document.getElementById("interactionVal").textContent = CONFIG.interactionStrength;
  });
  document.getElementById("particleCountSlider").addEventListener("input", (e) => {
    document.getElementById("particleCountSliderVal").textContent = e.target.value;
  });
  document.getElementById("applyParticles").addEventListener("click", async () => {
    const newCount = parseInt(document.getElementById("particleCountSlider").value, 10);
    await resizeParticles(newCount);
  });
  // document.getElementById("particleCount").textContent = CONFIG.numParticles;

  // Always-link size to physics
  particleSizeInput.addEventListener("input", async (e) => {
    const newPR = parseFloat(e.target.value);
    particleSizeVal.textContent = newPR.toFixed(3);
    applyParticleSize(newPR);
  });

  // Loop
  let last = performance.now();
  function frame() {
    const now = performance.now();
    let dt = (now - last) / 1000;
    last = now;

    if (!isPaused) {
      const maxDt = 1 / CONFIG.maxTimestepFPS;
      dt = Math.min(dt * CONFIG.timeScale, maxDt);

      const steps = dynamicSubsteps;                // adaptive substeps
      const dtSub = dt / steps;

      const encoder = device.createCommandEncoder();
      const computePass = encoder.beginComputePass();
      computePass.setBindGroup(0, computeBindGroup);

      for (let s = 0; s < steps; s++) {
        writeUniforms(dtSub);
        computePass.setPipeline(predictPipeline);
        computePass.dispatchWorkgroups(Math.ceil(CONFIG.numParticles / 256));

        computePass.setPipeline(clearPipeline);
        computePass.dispatchWorkgroups(Math.ceil(numCells / 256));

        computePass.setPipeline(buildGridPipeline);
        computePass.dispatchWorkgroups(Math.ceil(CONFIG.numParticles / 256));

        if (CONFIG.interactionStrength !== 0 && (mouseLeft || mouseRight)) {
          const cellSize = CONFIG.smoothingRadius * CELL_SCALE;
          const ring = Math.ceil(CONFIG.interactionRadius / cellSize);
          const side = 2 * ring + 1;
          const cellCount = side * side;
          computePass.setPipeline(applyMousePipeline);
          computePass.dispatchWorkgroups(cellCount);
        }

        computePass.setPipeline(densityPipeline);
        computePass.dispatchWorkgroups(Math.ceil(CONFIG.numParticles / 256));

        computePass.setPipeline(updatePipeline);
        computePass.dispatchWorkgroups(Math.ceil(CONFIG.numParticles / 256));
      }
      computePass.end();

      const texView = context.getCurrentTexture().createView();
      const renderPass = encoder.beginRenderPass({
        colorAttachments: [{ view: texView, loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }]
      });
      renderPass.setPipeline(renderPipeline);
      renderPass.setBindGroup(0, renderBindGroup);
      renderPass.setVertexBuffer(0, quadVB);
      renderPass.draw(6, CONFIG.numParticles, 0, 0);
      renderPass.end();

      device.queue.submit([encoder.finish()]);
    }

    requestAnimationFrame(frame);
  }
  frame();
}

initWebGPU().catch(err => {
  console.error(err);
  alert("WebGPU init failed: " + err.message);
});