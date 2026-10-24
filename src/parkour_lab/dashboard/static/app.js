/* AI Parkour Lab research dashboard.
 *
 * Plain ES modules + a locally vendored Three.js build.  No network access is required at
 * runtime: the page talks only to the same-origin FastAPI backend.
 */
import * as THREE from "/static/vendor/three.module.js";

// ---------------------------------------------------------------------------------------
// State and helpers
// ---------------------------------------------------------------------------------------
const state = {
  levels: [],
  runs: [],
  selectedRun: null,
  runDetail: null,
  jobId: null,
  jobStatus: null,
  jobLogOffset: 0,
  replay: null,
  replayIndex: 0,
  replayTime: 0,
  playing: true,
  playbackSpeed: 1,
  follow: true,
  levelViewer: null,
  replayViewer: null,
  charts: {},
  activeTab: "overview",
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

function el(tag, attributes = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2), value);
    } else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of Array.isArray(children) ? children : [children]) {
    if (child === null || child === undefined) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ? JSON.stringify(body.detail) : detail;
    } catch (error) {
      /* keep status text */
    }
    throw new Error(`${response.status} ${detail}`);
  }
  return response.json();
}

const fmt = {
  int: (value) => (value === null || value === undefined ? "–" : Number(value).toLocaleString()),
  num: (value, digits = 2) =>
    value === null || value === undefined || Number.isNaN(Number(value))
      ? "–"
      : Number(value).toFixed(digits),
  pct: (value, digits = 1) =>
    value === null || value === undefined ? "–" : `${(Number(value) * 100).toFixed(digits)}%`,
  date: (value) => (value ? new Date(value).toLocaleString() : "–"),
};

const COLORS = {
  accent: "#5aa9ff",
  green: "#5fd7a0",
  amber: "#f0b429",
  red: "#ef6461",
  purple: "#b592ff",
  grey: "#8b98a9",
};

function showBanner(message, tone = "info") {
  const banner = $("#banner");
  banner.textContent = message;
  banner.dataset.tone = tone;
  banner.classList.remove("hidden");
  if (tone !== "error") setTimeout(() => banner.classList.add("hidden"), 6000);
}

function hideBanner() {
  $("#banner").classList.add("hidden");
}

function setActiveTab(tab) {
  state.activeTab = tab;
  $$(".tab").forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
  $$(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === `panel-${tab}`));
  if (tab === "levels" && state.levelViewer) state.levelViewer.resize();
  if (tab === "replay" && state.replayViewer) state.replayViewer.resize();
}

// ---------------------------------------------------------------------------------------
// Charts (canvas, no dependencies)
// ---------------------------------------------------------------------------------------
class LineChart {
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.height = Number(canvas.getAttribute("height")) || 220;
    this.options = { yPercent: false, yMin: null, ...options };
    this.context = canvas.getContext("2d");
    this.series = [];
    this.hover = null;
    canvas.addEventListener("mousemove", (event) => {
      const rect = canvas.getBoundingClientRect();
      this.hover = { x: event.clientX - rect.left, y: event.clientY - rect.top };
      this.draw();
    });
    canvas.addEventListener("mouseleave", () => {
      this.hover = null;
      this.draw();
    });
    this._resizeObserver = new ResizeObserver(() => this.draw());
    this._resizeObserver.observe(canvas);
  }

  setSeries(series) {
    this.series = series.filter((entry) => entry.points && entry.points.length);
    this.draw();
  }

  resize() {
    this.draw();
  }

  draw() {
    const canvas = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(240, canvas.clientWidth);
    const height = this.height;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.height = `${height}px`;
    }
    const ctx = this.context;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const pad = { left: 48, right: 14, top: 26, bottom: 26 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const series of this.series) {
      for (const [x, y] of series.points) {
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      }
    }
    if (!Number.isFinite(minX)) {
      ctx.fillStyle = COLORS.grey;
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText(this.options.emptyText || "No data yet", pad.left, height / 2);
      return;
    }
    if (this.options.yPercent) {
      minY = Math.min(0, minY);
      maxY = Math.max(0.05, maxY);
    }
    if (this.options.yMin !== null) minY = Math.min(minY, this.options.yMin);
    if (maxX - minX < 1e-9) maxX = minX + 1;
    if (maxY - minY < 1e-9) { maxY += 0.5; minY -= 0.5; }
    const yPad = (maxY - minY) * 0.08;
    minY -= yPad; maxY += yPad;

    const sx = (x) => pad.left + ((x - minX) / (maxX - minX)) * plotW;
    const sy = (y) => pad.top + (1 - (y - minY) / (maxY - minY)) * plotH;

    // grid
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.lineWidth = 1;
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillStyle = COLORS.grey;
    for (let i = 0; i <= 4; i += 1) {
      const value = minY + ((maxY - minY) * i) / 4;
      const y = sy(value);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
      ctx.stroke();
      const label = this.options.yPercent ? `${(value * 100).toFixed(0)}%` : value.toFixed(2);
      ctx.fillText(label, 6, y + 3);
    }
    for (let i = 0; i <= 4; i += 1) {
      const value = minX + ((maxX - minX) * i) / 4;
      const x = sx(value);
      ctx.fillText(this.options.xFormat ? this.options.xFormat(value) : value.toFixed(0),
        x - 10, height - 8);
    }
    if (minY < 0 && maxY > 0) {
      ctx.strokeStyle = "rgba(255,255,255,0.18)";
      ctx.beginPath();
      ctx.moveTo(pad.left, sy(0));
      ctx.lineTo(width - pad.right, sy(0));
      ctx.stroke();
    }

    // series
    this.series.forEach((series, index) => {
      const color = series.color || Object.values(COLORS)[index % 5];
      ctx.strokeStyle = color;
      ctx.lineWidth = series.width || 1.8;
      ctx.beginPath();
      let started = false;
      for (const [x, y] of series.points) {
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        const px = sx(x), py = sy(y);
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke();
      if (series.fill) {
        const last = series.points[series.points.length - 1];
        const first = series.points[0];
        ctx.lineTo(sx(last[0]), sy(minY));
        ctx.lineTo(sx(first[0]), sy(minY));
        ctx.closePath();
        ctx.fillStyle = `${color}22`;
        ctx.fill();
      }
    });

    // legend
    ctx.font = "11px system-ui, sans-serif";
    let legendX = pad.left;
    this.series.forEach((series, index) => {
      const color = series.color || Object.values(COLORS)[index % 5];
      ctx.fillStyle = color;
      ctx.fillRect(legendX, 8, 8, 8);
      ctx.fillStyle = "#c7d2df";
      ctx.fillText(series.name, legendX + 12, 15);
      legendX += ctx.measureText(series.name).width + 30;
    });

    // hover readout
    if (this.hover && this.hover.x > pad.left && this.hover.x < width - pad.right) {
      const xValue = minX + ((this.hover.x - pad.left) / plotW) * (maxX - minX);
      ctx.strokeStyle = "rgba(255,255,255,0.2)";
      ctx.beginPath();
      ctx.moveTo(this.hover.x, pad.top);
      ctx.lineTo(this.hover.x, height - pad.bottom);
      ctx.stroke();
      const lines = this.series.map((series) => {
        const nearest = series.points.reduce(
          (best, point) => (Math.abs(point[0] - xValue) < Math.abs(best[0] - xValue) ? point : best),
          series.points[0],
        );
        return `${series.name}: ${this.options.yPercent ? fmt.pct(nearest[1]) : fmt.num(nearest[1], 3)}`;
      });
      ctx.fillStyle = "rgba(10,14,20,0.9)";
      const boxWidth = 170;
      const boxHeight = 12 + lines.length * 14;
      const boxX = Math.min(this.hover.x + 12, width - boxWidth - 6);
      ctx.fillRect(boxX, pad.top, boxWidth, boxHeight);
      ctx.strokeStyle = "rgba(255,255,255,0.1)";
      ctx.strokeRect(boxX, pad.top, boxWidth, boxHeight);
      ctx.fillStyle = "#dfe7f1";
      lines.forEach((line, index) => ctx.fillText(line, boxX + 7, pad.top + 15 + index * 14));
    }
  }
}

class BarChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.height = Number(canvas.getAttribute("height")) || 220;
    this.rows = [];
    this._resizeObserver = new ResizeObserver(() => this.draw());
    this._resizeObserver.observe(canvas);
  }

  setRows(rows) {
    this.rows = rows;
    this.draw();
  }

  draw() {
    const canvas = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(240, canvas.clientWidth);
    const height = this.height;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.height = `${height}px`;
    const ctx = this.context;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!this.rows.length) {
      ctx.fillStyle = COLORS.grey;
      ctx.font = "12px system-ui";
      ctx.fillText("Select two runs and press Compare", 16, height / 2);
      return;
    }
    const pad = { left: 42, right: 12, top: 26, bottom: 44 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    ctx.font = "10px ui-monospace, monospace";
    for (let i = 0; i <= 4; i += 1) {
      const y = pad.top + (plotH * i) / 4;
      ctx.strokeStyle = "rgba(255,255,255,0.07)";
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
      ctx.stroke();
      ctx.fillStyle = COLORS.grey;
      ctx.fillText(`${100 - i * 25}%`, 6, y + 3);
    }
    const group = plotW / this.rows.length;
    const barWidth = Math.max(4, Math.min(16, group * 0.3));
    this.rows.forEach((row, index) => {
      const baseX = pad.left + index * group + group / 2;
      const a = Math.max(0, Math.min(1, row.baseline_success));
      const b = Math.max(0, Math.min(1, row.candidate_success));
      ctx.fillStyle = COLORS.accent;
      ctx.fillRect(baseX - barWidth - 1, pad.top + plotH * (1 - a), barWidth, plotH * a);
      ctx.fillStyle = COLORS.green;
      ctx.fillRect(baseX + 1, pad.top + plotH * (1 - b), barWidth, plotH * b);
      ctx.save();
      ctx.translate(baseX, height - 26);
      ctx.rotate(-Math.PI / 5);
      ctx.fillStyle = COLORS.grey;
      ctx.textAlign = "right";
      ctx.fillText(row.level_id.replace("level-", "L"), 0, 0);
      ctx.restore();
    });
    ctx.textAlign = "left";
    ctx.fillStyle = COLORS.accent;
    ctx.fillRect(pad.left, 8, 8, 8);
    ctx.fillStyle = "#c7d2df";
    ctx.font = "11px system-ui";
    ctx.fillText("baseline", pad.left + 12, 15);
    ctx.fillStyle = COLORS.green;
    ctx.fillRect(pad.left + 84, 8, 8, 8);
    ctx.fillStyle = "#c7d2df";
    ctx.fillText("candidate", pad.left + 96, 15);
  }
}

// ---------------------------------------------------------------------------------------
// 3D scene
// ---------------------------------------------------------------------------------------
const BODY_GEOMETRY = {
  pelvis: { kind: "box", size: [0.27, 0.21, 0.17], offset: [0, 0, 0], color: 0x2f3947 },
  torso: { kind: "box", size: [0.33, 0.22, 0.43], offset: [0, 0, 0.22], color: 0x3f7fd4 },
  head: { kind: "sphere", size: [0.21, 0.21, 0.21], offset: [0, 0, 0.115], color: 0xd6a886 },
  thigh_l: { kind: "box", size: [0.13, 0.13, 0.40], offset: [0, 0, -0.20], color: 0x39424f },
  thigh_r: { kind: "box", size: [0.13, 0.13, 0.40], offset: [0, 0, -0.20], color: 0x39424f },
  shin_l: { kind: "box", size: [0.10, 0.10, 0.38], offset: [0, 0, -0.19], color: 0xd6a886 },
  shin_r: { kind: "box", size: [0.10, 0.10, 0.38], offset: [0, 0, -0.19], color: 0xd6a886 },
  foot_l: { kind: "box", size: [0.26, 0.11, 0.07], offset: [0, 0, -0.035], color: 0x1b1f26 },
  foot_r: { kind: "box", size: [0.26, 0.11, 0.07], offset: [0, 0, -0.035], color: 0x1b1f26 },
  upperarm_l: { kind: "box", size: [0.09, 0.09, 0.26], offset: [0, 0, -0.13], color: 0x3f7fd4 },
  upperarm_r: { kind: "box", size: [0.09, 0.09, 0.26], offset: [0, 0, -0.13], color: 0x3f7fd4 },
  forearm_l: { kind: "box", size: [0.075, 0.075, 0.24], offset: [0, 0, -0.12], color: 0xd6a886 },
  forearm_r: { kind: "box", size: [0.075, 0.075, 0.24], offset: [0, 0, -0.12], color: 0xd6a886 },
};

function mjToThree(position) {
  // MuJoCo is x-forward/y-left/z-up; Three.js is y-up. The scene group handles the
  // rotation, so this helper is only for camera targets in world space.
  return new THREE.Vector3(position[0], position[2], -position[1]);
}

class ParkourViewer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(48, 1, 0.05, 200);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.append(this.renderer.domElement);

    this.scene.add(new THREE.HemisphereLight(0xbdd4ff, 0x1a2230, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(6, 12, 8);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x8fb6ff, 0.6);
    rim.position.set(-8, 6, -7);
    this.scene.add(rim);

    // Everything in the level group is expressed in MuJoCo coordinates.
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;
    this.scene.add(this.world);

    this.grid = new THREE.GridHelper(60, 60, 0x2f3b4c, 0x1c2532);
    this.grid.position.y = 0;
    this.scene.add(this.grid);

    this.levelGroup = new THREE.Group();
    this.robotGroup = new THREE.Group();
    this.world.add(this.levelGroup, this.robotGroup);
    this.robotMeshes = {};
    this.moverMeshes = {};
    this.finishMarker = null;

    this.target = new THREE.Vector3(0, 1, 0);
    this.azimuth = -0.9;
    this.elevation = 0.5;
    this.distance = 7;
    this._clock = new THREE.Clock();
    this._bindControls();
    this.resize();
    this._resizeObserver = new ResizeObserver(() => this.resize());
    this._resizeObserver.observe(container);
    this._loop();
  }

  _bindControls() {
    const canvas = this.renderer.domElement;
    let dragging = null;
    canvas.addEventListener("pointerdown", (event) => {
      dragging = { x: event.clientX, y: event.clientY, pan: event.button === 2 || event.shiftKey };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointerup", (event) => {
      dragging = null;
      try { canvas.releasePointerCapture(event.pointerId); } catch (error) { /* ignore */ }
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      const dx = event.clientX - dragging.x;
      const dy = event.clientY - dragging.y;
      dragging.x = event.clientX;
      dragging.y = event.clientY;
      if (dragging.pan) {
        const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
        const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
        this.target.addScaledVector(right, -dx * this.distance * 0.0016);
        this.target.addScaledVector(up, dy * this.distance * 0.0016);
      } else {
        this.azimuth -= dx * 0.008;
        this.elevation = Math.min(1.45, Math.max(-0.2, this.elevation + dy * 0.006));
      }
    });
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.distance = Math.min(30, Math.max(1.5, this.distance * (1 + Math.sign(event.deltaY) * 0.1)));
    }, { passive: false });
  }

  resize() {
    const width = Math.max(280, this.container.clientWidth);
    const height = Math.max(220, this.container.clientHeight || 360);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  loadLevel(level) {
    this.levelGroup.clear();
    this.robotMeshes = {};
    this.moverMeshes = {};
    this.finishMarker = null;
    this.robotGroup.clear();
    if (!level) return;
    const palette = {
      platform: 0x59626f,
      step: 0x6b7686,
      wall: 0x8a94a6,
      rail: 0xd3a233,
      rooftop: 0x46586e,
      spike_strip: 0xb4392f,
      moving_platform: 0x2f7f9e,
    };
    for (const obstacle of level.obstacles || []) {
      if (obstacle.kind === "gap") continue;
      const [x, y, z] = obstacle.position;
      const [sx, sy, sz] = obstacle.size;
      const color = obstacle.hazard ? 0xb4392f : palette[obstacle.kind] || 0x59626f;
      const geometry = new THREE.BoxGeometry(sx * 2, sy * 2, sz * 2);
      const material = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.82,
        metalness: 0.05,
        emissive: obstacle.hazard ? 0x2a0705 : 0x000000,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(x, y, z);
      this.levelGroup.add(mesh);
      if (obstacle.kind === "moving_platform") this.moverMeshes[obstacle.id] = mesh;
    }
    // finish marker: a glowing ring plus a soft column
    const finish = level.finish;
    if (finish) {
      const [fx, fy, fz] = finish.position;
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(finish.radius, 0.035, 10, 40),
        new THREE.MeshStandardMaterial({ color: 0x5fd7a0, emissive: 0x2c8f68, roughness: 0.4 }),
      );
      ring.rotation.x = Math.PI / 2;
      ring.position.set(fx, fy, fz - 0.9);
      this.levelGroup.add(ring);
      const beam = new THREE.Mesh(
        new THREE.CylinderGeometry(finish.radius * 0.9, finish.radius * 0.9, 2.4, 28, 1, true),
        new THREE.MeshBasicMaterial({
          color: 0x5fd7a0, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false,
        }),
      );
      beam.position.set(fx, fy, fz);
      this.levelGroup.add(beam);
      this.finishMarker = ring;
    }
    const spawn = level.spawn.position;
    const disc = new THREE.Mesh(
      new THREE.CylinderGeometry(0.45, 0.45, 0.02, 24),
      new THREE.MeshBasicMaterial({ color: 0x5aa9ff, transparent: true, opacity: 0.45 }),
    );
    disc.position.set(spawn[0], spawn[1], spawn[2] - 0.9);
    this.levelGroup.add(disc);
    this.resetCamera(level);
    this.grid.position.y = 0;
  }

  resetCamera(level = null) {
    if (level) {
      const start = level.spawn.position;
      const finish = level.finish.position;
      this.target = mjToThree([(start[0] + finish[0]) / 2, (start[1] + finish[1]) / 2, 0.8]);
      this.distance = Math.max(5, Math.abs(finish[0] - start[0]) * 0.9 + 2);
    } else {
      this.target.set(0, 0.9, 0);
      this.distance = 7;
    }
    this.azimuth = -0.9;
    this.elevation = 0.42;
  }

  _robotMesh(name) {
    if (this.robotMeshes[name]) return this.robotMeshes[name];
    const spec = BODY_GEOMETRY[name];
    if (!spec) return null;
    const group = new THREE.Group();
    const material = new THREE.MeshStandardMaterial({
      color: spec.color, roughness: 0.62, metalness: 0.08,
    });
    let geometry;
    if (spec.kind === "sphere") {
      geometry = new THREE.SphereGeometry(spec.size[0] / 2, 18, 14);
    } else {
      geometry = new THREE.BoxGeometry(spec.size[0], spec.size[1], spec.size[2]);
    }
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(spec.offset[0], spec.offset[1], spec.offset[2]);
    group.add(mesh);
    this.robotGroup.add(group);
    this.robotMeshes[name] = group;
    return group;
  }

  setPose(bodies, movers = null) {
    if (!bodies) return;
    for (const [name, values] of Object.entries(bodies)) {
      const group = this._robotMesh(name);
      if (!group) continue;
      group.visible = true;
      group.position.set(values[0], values[1], values[2]);
      group.quaternion.set(values[4], values[5], values[6], values[3]);
    }
    if (movers) {
      for (const [id, position] of Object.entries(movers)) {
        const mesh = this.moverMeshes[id];
        if (mesh) mesh.position.set(position[0], position[1], position[2]);
      }
    }
  }

  clearRobot() {
    Object.values(this.robotMeshes).forEach((group) => { group.visible = false; });
  }

  focus(position) {
    const target = mjToThree(position);
    this.target.lerp(target, 0.15);
  }

  _loop() {
    const animate = () => {
      requestAnimationFrame(animate);
      const phi = Math.PI / 2 - this.elevation;
      this.camera.position.set(
        this.target.x + this.distance * Math.sin(phi) * Math.cos(this.azimuth),
        this.target.y + this.distance * Math.cos(phi),
        this.target.z + this.distance * Math.sin(phi) * Math.sin(this.azimuth),
      );
      this.camera.lookAt(this.target);
      if (this.finishMarker) {
        const t = performance.now() / 1000;
        this.finishMarker.material.emissiveIntensity = 0.6 + 0.4 * Math.sin(t * 2);
      }
      this.renderer.render(this.scene, this.camera);
    };
    animate();
  }
}

// ---------------------------------------------------------------------------------------
// Overview
// ---------------------------------------------------------------------------------------
async function loadHealth() {
  try {
    const health = await api("/api/health");
    $("#health-dot").className = "dot ok";
    $("#health-text").textContent = `connected · ${health.cpu_count} CPU`;
    $("#about-versions").textContent =
      `backend ${health.version} · python ${health.python} · numpy ${health.numpy} · ` +
      `mujoco ${health.mujoco} · runs: ${health.runs_dir}`;
    $("#footer-runs").textContent = health.runs_dir;
    if (health.active_job) {
      state.jobId = health.active_job.job_id;
      state.jobStatus = health.active_job.status;
      startJobPolling();
    }
  } catch (error) {
    $("#health-dot").className = "dot bad";
    $("#health-text").textContent = "backend unreachable";
    showBanner(`Backend unreachable: ${error.message}`, "error");
  }
}

async function loadRuns() {
  try {
    const data = await api("/api/runs");
    state.runs = data.runs || [];
    renderRunList();
    renderRunSelectors();
    if (!state.selectedRun && state.runs.length) {
      selectRun(state.runs[0].run_id);
    } else if (state.selectedRun && !state.runs.some((run) => run.run_id === state.selectedRun)) {
      state.selectedRun = null;
      $("#run-empty").classList.remove("hidden");
      $("#run-detail").classList.add("hidden");
    }
  } catch (error) {
    showBanner(`Could not load runs: ${error.message}`, "error");
  }
}

function renderRunList() {
  const list = $("#run-list");
  list.innerHTML = "";
  if (!state.runs.length) {
    list.append(el("p", { class: "muted", text: "No runs yet — start one from the Training tab." }));
    return;
  }
  for (const run of state.runs) {
    const card = el("button", {
      class: `run-item${run.run_id === state.selectedRun ? " active" : ""}`,
      onclick: () => selectRun(run.run_id),
    }, [
      el("div", { class: "rid", text: run.run_id }),
      el("div", { class: "row" }, [
        el("span", { text: run.status }),
        el("span", {
          text: run.progress_steps
            ? `${fmt.int(run.progress_steps)} / ${fmt.int(run.target_steps)} steps`
            : `${fmt.int(run.total_steps)} steps`,
        }),
        el("span", { text: fmt.pct(run.best_eval_success) }),
      ]),
      el("div", { class: "row" }, [
        el("span", { text: `${(run.level_ids || []).length} level(s)` }),
        el("span", { text: run.curriculum || "" }),
        el("span", { text: run.updated_utc ? new Date(run.updated_utc).toLocaleTimeString() : "" }),
      ]),
    ]);
    list.append(card);
  }
}

async function selectRun(runId) {
  state.selectedRun = runId;
  renderRunList();
  try {
    const detail = await api(`/api/runs/${encodeURIComponent(runId)}`);
    state.runDetail = detail;
    $("#run-empty").classList.add("hidden");
    $("#run-detail").classList.remove("hidden");
    renderRunDetail(detail);
  } catch (error) {
    showBanner(`Could not load run ${runId}: ${error.message}`, "error");
  }
}

function kv(label, value) {
  return el("div", { class: "kv" }, [
    el("span", { class: "k", text: label }),
    el("span", { class: "v", text: value }),
  ]);
}

function stat(label, value, hint = "") {
  return el("div", { class: "stat" }, [
    el("div", { class: "label", text: label }),
    el("div", { class: "value", text: value }),
    hint ? el("div", { class: "hint", text: hint }) : null,
  ]);
}

function renderRunDetail(detail) {
  const summary = detail.summary || {};
  const config = detail.config || {};
  const aggregate = detail.episode_aggregate || { levels: [], terminations: {} };
  const evaluations = detail.evaluation_history || [];
  const latestEval = detail.latest_evaluation;
  const latestOverall = latestEval?.aggregate?.overall || {};

  $("#detail-title").textContent = detail.run_id;
  const badge = $("#detail-status");
  badge.textContent = summary.status || "unknown";
  badge.className = `pill ${summary.status === "completed" ? "ok" : summary.status === "running" ? "warn" : ""}`;

  $("#detail-stats").innerHTML = "";
  $("#detail-stats").append(
    stat("Best eval success", fmt.pct(summary.best_eval_success), `iteration ${summary.best_eval_iteration ?? "–"}`),
    stat("Latest eval success", fmt.pct(latestOverall.success_rate), `${latestOverall.episodes ?? 0} episodes`),
    stat("Latest completion", fmt.pct(latestOverall.mean_completion), "mean distance to finish"),
    stat(
      "Training steps",
      fmt.int(rows.length ? rows[rows.length - 1].total_steps : summary.total_steps),
      `of ${fmt.int(config.total_steps)} · ${fmt.int(config.num_envs)} envs`,
    ),
    stat("Episodes", fmt.int(aggregate.episodes), `${Object.keys(aggregate.terminations || {}).length} end reasons`),
    stat("Levels", fmt.int((config.level_ids || []).length), config.curriculum || ""),
  );

  $("#detail-meta").innerHTML = "";
  $("#detail-meta").append(
    kv("run id", detail.run_id),
    kv("created", fmt.date(summary.created_utc)),
    kv("envs / workers", `${config.num_envs ?? "–"} / ${config.workers ?? "–"}`),
    kv("seed", String(config.seed ?? "–")),
    kv("action scale / smoothing", `${config.action_scale ?? "–"} / ${config.action_smoothing ?? "–"}`),
    kv("control", `${config.control_decimation ?? "–"} × ${config.physics_timestep ?? "–"} s`),
    kv("actuation", config.actuation || "position"),
    kv("git", (detail.provenance?.git?.commit || "").slice(0, 12) || "–"),
  );

  // charts
  const rows = detail.training || [];
  const xs = rows.map((row) => row.total_steps);
  if (!state.charts.progress) state.charts.progress = new LineChart($("#chart-progress"));
  state.charts.progress.setSeries([
    { name: "episode return", color: COLORS.accent, points: rows.map((r, i) => [xs[i], r.episode_return]) },
    { name: "success ×100", color: COLORS.green, points: rows.map((r, i) => [xs[i], r.success_rate * 100]) },
  ]);

  if (!state.charts.eval) state.charts.eval = new LineChart($("#chart-eval"), { yPercent: true, yMin: 0 });
  state.charts.eval.setSeries([
    {
      name: "eval success",
      color: COLORS.green,
      fill: true,
      points: evaluations.map((row) => [row.total_steps, row.success_rate]),
    },
    {
      name: "eval completion",
      color: COLORS.grey,
      width: 1.2,
      points: evaluations.map((row) => [row.total_steps, row.mean_completion]),
    },
  ]);

  if (!state.charts.timeline) state.charts.timeline = new LineChart($("#chart-timeline"), { yPercent: true });
  const timeline = (aggregate.timeline || []).map((bucket) => [bucket.iteration, bucket.success_rate]);
  state.charts.timeline.setSeries([
    { name: "train success", color: COLORS.accent, points: timeline, fill: true },
  ]);

  // per-level bars
  const bars = $("#level-bars");
  bars.innerHTML = "";
  const levels = aggregate.levels || [];
  if (!levels.length) {
    bars.append(el("p", { class: "muted", text: "No episodes recorded yet." }));
  }
  for (const level of levels.slice(0, 24)) {
    const fill = Math.max(0, Math.min(1, level.success_rate));
    bars.append(
      el("div", { class: "bar-row" }, [
        el("span", { class: "name", title: level.level_id, text: level.level_id }),
        el("span", { class: "bar-track" }, [
          el("span", {
            class: `bar-fill${fill > 0.5 ? " good" : ""}`,
            style: `width:${(fill * 100).toFixed(1)}%`,
          }),
        ]),
        el("span", { class: "val", text: fmt.pct(fill) }),
      ]),
    );
    bars.append(
      el("div", { class: "bar-row" }, [
        el("span"),
        el("span", {
          class: "muted small",
          text: `${fmt.int(level.episodes)} episodes · mean completion ${fmt.pct(level.mean_completion)}`,
        }),
        el("span"),
      ]),
    );
  }

  // terminations
  const terms = aggregate.terminations || {};
  const total = Object.values(terms).reduce((sum, value) => sum + value, 0);
  const termList = $("#termination-list");
  termList.innerHTML = "";
  const colors = {
    course_complete: COLORS.green,
    timeout: COLORS.accent,
    fell_over: COLORS.red,
    collapsed: COLORS.amber,
    void_fall: COLORS.purple,
    hazard_contact: "#e05c8a",
    out_of_bounds: "#e08a4a",
  };
  if (!total) termList.append(el("p", { class: "muted", text: "No episodes yet." }));
  for (const [reason, count] of Object.entries(terms).sort((a, b) => b[1] - a[1])) {
    const share = count / total;
    termList.append(
      el("div", { class: "term-row" }, [
        el("span", {}, [
          el("span", {
            class: "swatch",
            style: `background:${colors[reason] || COLORS.grey}`,
          }),
          reason,
        ]),
        el("span", { class: "bar-track" }, [
          el("span", {
            class: "bar-fill",
            style: `width:${(share * 100).toFixed(1)}%;background:${colors[reason] || COLORS.grey}`,
          }),
        ]),
        el("span", { class: "val", text: fmt.int(count) }),
      ]),
    );
  }

  // stability readout
  const last = rows[rows.length - 1] || {};
  const unstable = Math.abs(last.approx_kl || 0) > 0.04 || (last.value_loss || 0) > 1e4;
  const stability = $("#stability");
  stability.innerHTML = "";
  stability.append(
    el("div", { class: "kv" }, [
      el("span", { class: "k", text: "value loss" }),
      el("span", { class: "v", text: fmt.num(last.value_loss, 4) }),
    ]),
    el("div", { class: "kv" }, [
      el("span", { class: "k", text: "approx KL" }),
      el("span", { class: "v", text: fmt.num(last.approx_kl, 4) }),
    ]),
    el("div", { class: "kv" }, [
      el("span", { class: "k", text: "clip fraction" }),
      el("span", { class: "v", text: fmt.num(last.clip_fraction, 3) }),
    ]),
    el("div", { class: "kv" }, [
      el("span", { class: "k", text: "explained var" }),
      el("span", { class: "v", text: fmt.num(last.explained_variance, 3) }),
    ]),
    el("div", { class: "kv" }, [
      el("span", { class: "k", text: "state" }),
      el("span", { class: "v", text: unstable ? "⚠ update instability" : "ok" }),
    ]),
    el("div", { class: "kv" }, [
      el("span", { class: "k", text: "throughput" }),
      el("span", { class: "v", text: `${fmt.num(last.steps_per_second, 0)} steps/s` }),
    ]),
  );

  // artifacts
  const artifacts = $("#artifacts");
  artifacts.innerHTML = "";
  const checkpoints = detail.checkpoints || [];
  const replays = detail.replays || [];
  artifacts.append(
    el("div", { class: "artifact-group" }, [
      el("h4", { text: `checkpoints (${checkpoints.length})` }),
      el("ul", {}, checkpoints.slice(-8).map((row) => el("li", { text: row.name }))),
    ]),
    el("div", { class: "artifact-group" }, [
      el("h4", { text: `replays (${replays.length})` }),
      el("ul", {}, replays.slice(-8).map((row) => el("li", { text: row.name }))),
    ]),
  );

  if (!state.charts.progress) state.charts.progress.resize();
}

// ---------------------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------------------
function renderTrainingLevels() {
  const picker = $("#train-levels");
  picker.innerHTML = "";
  for (const level of state.levels) {
    const input = el("input", { type: "checkbox", value: level.id });
    if (level.split !== "train") input.disabled = true;
    picker.append(
      el("label", {}, [
        input,
        el("span", { text: level.id }),
        el("span", { class: `split ${level.split === "held-out" ? "held" : ""}`, text: level.split }),
      ]),
    );
  }
}

async function startTraining(event) {
  event.preventDefault();
  const form = event.target;
  const levels = $$("#train-levels input:checked").map((input) => input.value);
  const payload = {
    levels,
    steps: Number(form.steps.value),
    envs: Number(form.envs.value),
    workers: Number(form.workers.value),
    curriculum: form.curriculum.value,
    seed: Number(form.seed.value),
    name: form.name.value || "",
    notes: form.notes.value || "",
    max_minutes: form.max_minutes.value ? Number(form.max_minutes.value) : null,
    rollout_steps: Number(form.rollout_steps.value) || 512,
    hidden_dim: Number(form.hidden_dim.value) || 128,
    eval_interval: Number(form.eval_interval.value) || 20,
    checkpoint_interval: Number(form.checkpoint_interval.value) || 20,
    settings: {},
    reward: {},
  };
  const numeric = (name, fallback) => {
    const raw = form[name]?.value;
    return raw === undefined || raw === "" ? fallback : Number(raw);
  };
  for (const name of [
    "physics_timestep",
    "control_decimation",
    "action_scale",
    "action_smoothing",
    "max_episode_seconds",
    "stagnation_seconds",
    "reward_ramp_fraction",
  ]) {
    const value = numeric(name, null);
    if (value !== null && Number.isFinite(value)) payload.settings[name] = value;
  }
  for (const field of [
    "progress_per_metre",
    "forward_velocity_per_second",
    "gait_per_second",
    "upright_per_second",
    "height_per_second",
    "alive_per_second",
    "fall_penalty",
    "action_rate_penalty",
  ]) {
    const value = numeric(`reward_${field}`, null);
    if (value !== null && Number.isFinite(value)) payload.reward[field] = value;
  }
  if (payload.curriculum === "single" && levels.length !== 1) {
    showBanner("The single-level curriculum needs exactly one selected level.", "error");
    return;
  }
  try {
    const job = await api("/api/train", { method: "POST", body: JSON.stringify(payload) });
    state.jobId = job.job_id;
    state.jobStatus = job.status;
    state.jobLogOffset = 0;
    $("#job-log").textContent = "";
    showBanner(`Training started: ${job.run_id}`, "info");
    startJobPolling();
  } catch (error) {
    showBanner(`Could not start training: ${error.message}`, "error");
  }
}

function startJobPolling() {
  if (!state.jobId) return;
  if (state.jobTimer) clearInterval(state.jobTimer);
  state.jobTimer = setInterval(pollJob, 1800);
  pollJob();
  $("#start-btn").disabled = true;
  $("#stop-btn").disabled = false;
}

async function pollJob() {
  if (!state.jobId) return;
  try {
    const data = await api(`/api/jobs/${state.jobId}/log?offset=${state.jobLogOffset}`);
    const job = data.job;
    state.jobStatus = job.status;
    const badge = $("#job-status");
    badge.textContent = job.status;
    badge.className = `pill ${job.status === "running" ? "warn" : job.status === "finished" ? "ok" : job.status === "failed" ? "bad" : ""}`;
    $("#job-meta").innerHTML = "";
    $("#job-meta").append(
      kv("job", job.job_id),
      kv("run", job.run_id),
      kv("status", job.status),
      kv("exit code", String(job.exit_code ?? "–")),
      kv("command", job.command.join(" ")),
    );
    if (data.lines.length) {
      const log = $("#job-log");
      const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 40;
      log.textContent += `${data.lines.join("\n")}\n`;
      if (atBottom) log.scrollTop = log.scrollHeight;
      state.jobLogOffset = data.offset;
    }
    if (job.status !== "running" && job.status !== "starting") {
      clearInterval(state.jobTimer);
      state.jobTimer = null;
      $("#start-btn").disabled = false;
      $("#stop-btn").disabled = true;
      await loadRuns();
      showBanner(`Training ${job.status}.`, job.status === "finished" ? "info" : "error");
    }
  } catch (error) {
    clearInterval(state.jobTimer);
    state.jobTimer = null;
    $("#start-btn").disabled = false;
    $("#stop-btn").disabled = true;
    showBanner(`Lost contact with the training job: ${error.message}`, "error");
  }
}

async function stopTraining() {
  if (!state.jobId) return;
  try {
    await api(`/api/jobs/${state.jobId}/stop`, { method: "POST" });
    await pollJob();
  } catch (error) {
    showBanner(`Could not stop training: ${error.message}`, "error");
  }
}

async function runBenchmark() {
  const button = $("#bench-btn");
  button.disabled = true;
  $("#bench-results").textContent = "measuring...";
  try {
    const data = await api("/api/benchmark?envs=1,2,4&steps=150");
    $("#bench-results").innerHTML = "";
    for (const row of data.results) {
      $("#bench-results").append(
        el("div", {
          text: `${row.num_envs} env · ${Math.round(row.steps_per_second).toLocaleString()} transitions/s · ${row.workers} worker(s)`,
        }),
      );
    }
  } catch (error) {
    $("#bench-results").textContent = `benchmark failed: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

// ---------------------------------------------------------------------------------------
// Levels
// ---------------------------------------------------------------------------------------
function renderLevelList() {
  const list = $("#level-list");
  list.innerHTML = "";
  $("#level-count").textContent = `${state.levels.length} fixed levels`;
  for (const level of state.levels) {
    list.append(
      el("button", {
        class: `level-item${level.id === state.selectedLevel ? " active" : ""}`,
        "data-level": level.id,
        onclick: () => selectLevel(level.id),
      }, [
        el("span", { class: "lid", text: `${level.order}. ${level.id}` }),
        el("span", { class: "diff", text: `d${level.difficulty} · ${level.split}` }),
      ]),
    );
  }
  if (state.levels.length) selectLevel(state.selectedLevel || state.levels[0].id);
}

function selectLevel(levelId) {
  const level = state.levels.find((entry) => entry.id === levelId);
  if (!level) return;
  state.selectedLevel = levelId;
  $$(".level-item").forEach((item) =>
    item.classList.toggle("active", item.dataset.level === levelId),
  );
  $("#level-title").textContent = level.title;
  $("#level-badge").textContent = `${level.id} · difficulty ${level.difficulty}`;
  $("#level-summary").textContent = level.summary || "";
  $("#level-details").innerHTML = "";
  $("#level-details").append(
    kv("length", `${fmt.num(level.finish.position[0] - level.spawn.position[0])} m`),
    kv("time limit", `${fmt.num(level.time_limit_seconds)} s`),
    kv("par time", level.par_time_seconds ? `${fmt.num(level.par_time_seconds)} s` : "–"),
    kv("split", level.split),
    kv("finish radius", `${fmt.num(level.finish.radius)} m`),
    kv("obstacles", String(level.obstacles.length)),
    kv("spawn", level.spawn.position.map((v) => v.toFixed(2)).join(", ")),
  );
  const rows = level.obstacles
    .map(
      (obstacle) => `<tr>
        <td>${obstacle.id}</td>
        <td>${obstacle.kind}</td>
        <td class="num">${obstacle.position.map((v) => v.toFixed(2)).join(", ")}</td>
        <td class="num">${obstacle.size.map((v) => v.toFixed(2)).join(", ")}</td>
        <td>${obstacle.hazard ? "hazard" : ""}${obstacle.motion ? ` moving ${obstacle.motion.amplitude} m / ${obstacle.motion.period_seconds} s` : ""}</td>
      </tr>`,
    )
    .join("");
  $("#level-obstacles").innerHTML =
    `<table><thead><tr><th>id</th><th>kind</th><th>position</th><th>half-size</th><th>notes</th></tr></thead><tbody>${rows}</tbody></table>`;
  if (!state.levelViewer) state.levelViewer = new ParkourViewer($("#level-viewer"));
  state.levelViewer.loadLevel(level);
  state.levelViewer.clearRobot();
}

// ---------------------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------------------
function renderRunSelectors() {
  const options = state.runs
    .map((run) => ({ value: run.run_id, label: `${run.run_id} (${fmt.pct(run.best_eval_success)})` }));
  for (const id of ["#replay-run", "#compare-a", "#compare-b"]) {
    const select = $(id);
    const previous = select.value;
    select.innerHTML = "";
    if (!options.length) {
      select.append(el("option", { value: "", text: "no runs" }));
      continue;
    }
    for (const option of options) select.append(el("option", { value: option.value, text: option.label }));
    if (options.some((option) => option.value === previous)) select.value = previous;
  }
  if (state.runs.length > 1) {
    $("#compare-a").value = state.runs[Math.min(1, state.runs.length - 1)].run_id;
    $("#compare-b").value = state.runs[0].run_id;
  }
  refreshReplayOptions();
}

async function refreshReplayOptions() {
  const runId = $("#replay-run").value;
  const select = $("#replay-file");
  select.innerHTML = "";
  if (!runId) {
    select.append(el("option", { value: "", text: "no run" }));
    return;
  }
  try {
    const data = await api(`/api/runs/${encodeURIComponent(runId)}/replays`);
    const replays = data.replays || [];
    if (!replays.length) {
      select.append(el("option", { value: "", text: "no replays recorded" }));
      $("#replay-info").textContent = "";
      return;
    }
    for (const replay of replays) {
      select.append(el("option", { value: replay.name, text: replay.name }));
    }
    select.value = replays[replays.length - 1].name;
  } catch (error) {
    select.append(el("option", { value: "", text: "unavailable" }));
  }
}

async function loadReplay() {
  const runId = $("#replay-run").value;
  const name = $("#replay-file").value;
  if (!runId || !name) {
    showBanner("No replay selected. Replays are recorded during evaluation.", "error");
    return;
  }
  try {
    const replay = await api(
      `/api/runs/${encodeURIComponent(runId)}/replay/${encodeURIComponent(name)}`,
    );
    state.replay = replay;
    state.replayIndex = 0;
    state.replayTime = 0;
    state.playing = true;
    state.playbackSpeed = Number($("#speed").value) || 1;
    $("#play-btn").textContent = "Pause";
    $("#scrub").max = String(Math.max(0, replay.frames.length - 1));
    $("#scrub").value = "0";
    if (!state.replayViewer) state.replayViewer = new ParkourViewer($("#replay-viewer"));
    state.replayViewer.loadLevel(replay.level);
    state.replayViewer.clearRobot();
    $("#replay-title").textContent = `${replay.level_title || replay.level_id}`;
    const badge = $("#replay-badge");
    badge.textContent = replay.success ? "success" : replay.termination_reason || "ended";
    badge.className = `pill ${replay.success ? "ok" : "warn"}`;
    $("#replay-info").innerHTML = "";
    $("#replay-info").append(
      kv("run", replay.run_id || runId),
      kv("level", replay.level_id),
      kv("seed", String(replay.seed ?? "–")),
      kv("result", replay.success ? "course complete" : replay.termination_reason || "ended"),
      kv("completion", fmt.pct(replay.completion_fraction)),
      kv("return", fmt.num(replay.episode_return)),
      kv("frames", String(replay.frames.length)),
    );
    applyReplayFrame(0);
    hideBanner();
  } catch (error) {
    showBanner(`Could not load replay: ${error.message}`, "error");
  }
}

function applyReplayFrame(index) {
  const replay = state.replay;
  if (!replay || !replay.frames.length) return;
  const clamped = Math.max(0, Math.min(replay.frames.length - 1, Math.round(index)));
  const frame = replay.frames[clamped];
  if (!frame) return;
  state.replayViewer.setPose(frame.bodies, frame.movers || {});
  if (state.follow && frame.bodies.pelvis) state.replayViewer.focus(frame.bodies.pelvis);
  state.replayIndex = clamped;
  const duration = replay.frames[replay.frames.length - 1].t || 0;
  $("#replay-time").textContent = `${(frame.t || 0).toFixed(2)}s / ${duration.toFixed(2)}s`;
  $("#scrub").value = String(clamped);
}

function tickPlayback(deltaSeconds) {
  const replay = state.replay;
  if (!replay || !state.playing || state.activeTab !== "replay") return;
  const frames = replay.frames;
  if (!frames.length) return;
  state.replayTime += deltaSeconds * state.playbackSpeed;
  const start = frames[state.replayIndex]?.t || 0;
  let index = state.replayIndex;
  while (index + 1 < frames.length && frames[index + 1].t <= start + state.replayTime) index += 1;
  if (index >= frames.length - 1) {
    state.playing = false;
    $("#play-btn").textContent = "Play";
  }
  if (index !== state.replayIndex) {
    state.replayTime = 0;
    applyReplayFrame(index);
  }
}

// ---------------------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------------------
let lastComparison = null;

async function compareRuns() {
  const a = $("#compare-a").value;
  const b = $("#compare-b").value;
  if (!a || !b) {
    showBanner("Select two runs with saved evaluations.", "error");
    return;
  }
  if (a === b) {
    showBanner("Pick two different runs to compare.", "error");
    return;
  }
  try {
    const comparison = await api(`/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
    lastComparison = comparison;
    const overall = comparison.overall;
    $("#compare-summary").innerHTML = "";
    $("#compare-summary").append(
      stat("A success", fmt.pct(overall.baseline_success_rate), a),
      stat("B success", fmt.pct(overall.candidate_success_rate), b),
      stat("Δ success", `${(overall.delta_success_rate * 100).toFixed(1)} pp`, "candidate − baseline"),
      stat("Δ completion", `${(overall.delta_mean_completion * 100).toFixed(1)} pp`, "candidate − baseline"),
    );
    const rows = comparison.rows
      .map(
        (row) => `<tr>
          <td>${row.level_id}</td>
          <td class="num">${fmt.pct(row.baseline_success)}</td>
          <td class="num">${fmt.pct(row.candidate_success)}</td>
          <td class="num" style="color:${row.delta_success >= 0 ? COLORS.green : COLORS.red}">${(row.delta_success * 100).toFixed(1)} pp</td>
          <td class="num">${fmt.pct(row.baseline_completion)}</td>
          <td class="num">${fmt.pct(row.candidate_completion)}</td>
          <td class="num">${row.episodes}</td>
        </tr>`,
      )
      .join("");
    $("#compare-table").innerHTML =
      `<table><thead><tr><th>level</th><th>A success</th><th>B success</th><th>Δ</th><th>A completion</th><th>B completion</th><th>episodes</th></tr></thead><tbody>${rows}</tbody></table>`;
    if (!state.charts.compare) state.charts.compare = new BarChart($("#chart-compare"));
    state.charts.compare.setRows(comparison.rows);
    hideBanner();
  } catch (error) {
    showBanner(`Comparison failed: ${error.message}`, "error");
  }
}

// ---------------------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------------------
function bindEvents() {
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => setActiveTab(tab.dataset.tab)));
  $("#refresh-runs").addEventListener("click", () => loadRuns());
  $("#train-form").addEventListener("submit", startTraining);
  $("#stop-btn").addEventListener("click", stopTraining);
  $("#bench-btn").addEventListener("click", runBenchmark);
  $("#replay-run").addEventListener("change", refreshReplayOptions);
  $("#replay-load").addEventListener("click", loadReplay);
  $("#compare-btn").addEventListener("click", compareRuns);
  $("#play-btn").addEventListener("click", () => {
    if (!state.replay) return;
    state.playing = !state.playing;
    $("#play-btn").textContent = state.playing ? "Pause" : "Play";
  });
  $("#scrub").addEventListener("input", (event) => {
    state.playing = false;
    $("#play-btn").textContent = "Play";
    applyReplayFrame(Number(event.target.value));
  });
  $("#speed").addEventListener("change", (event) => {
    state.playbackSpeed = Number(event.target.value) || 1;
  });
  $("#follow").addEventListener("change", (event) => {
    state.follow = event.target.checked;
  });
  $("#reset-cam").addEventListener("click", () => {
    if (state.replayViewer && state.replay) state.replayViewer.resetCamera(state.replay.level);
  });
  window.addEventListener("resize", () => {
    Object.values(state.charts).forEach((chart) => chart.resize());
  });
  let last = performance.now();
  const loop = (now) => {
    const delta = Math.min((now - last) / 1000, 0.25);
    last = now;
    tickPlayback(delta);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

async function init() {
  bindEvents();
  const hash = (location.hash || "#overview").slice(1);
  if (["overview", "training", "levels", "replay", "compare", "about"].includes(hash)) {
    setActiveTab(hash);
  }
  await loadHealth();
  try {
    const data = await api("/api/levels");
    state.levels = data.levels || [];
  } catch (error) {
    showBanner(`Could not load levels: ${error.message}`, "error");
  }
  renderTrainingLevels();
  renderLevelList();
  await loadRuns();
  refreshReplayOptions();
  setInterval(loadHealth, 30000);
  if (!state.jobId) loadRuns();
  setInterval(() => { if (!state.jobTimer) loadRuns(); }, 20000);
}

init();
