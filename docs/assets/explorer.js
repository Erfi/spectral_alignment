// Interactive explorer: Q-function / SR heatmaps in four rooms as a function of the action repeat k.
// Data comes from notebooks/cover_figure.ipynb ("Website" cells) -> docs/assets/heatmaps.js (window.HEATMAP_DATA).
// Each map is a base64 uint8 array of size res*res: 0..254 = value scaled to the map's own range, 255 = wall.
(function () {
  "use strict";

  const WALL = 255;
  const WALL_RGB = [230, 230, 230];
  // matplotlib "Blues", the colormap of the paper figures
  const BLUES = ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"];

  const state = { data: null, kIndex: 0, quantity: "q", u: null, v: null };
  const els = {
    slider: document.getElementById("k-slider"),
    kOut: document.getElementById("k-out"),
    ticks: document.getElementById("k-ticks-labels"),
    quantityButtons: document.querySelectorAll("#explorer .segmented button"),
    status: document.getElementById("status"),
    bars: document.getElementById("bars"),
    markerRole: document.getElementById("marker-role"),
  };
  // smooth = bilinear upscaling for the continuous maps, nearest for the grid world
  const panels = [
    { key: "discrete", canvas: document.getElementById("disc-canvas"), smooth: false },
    { key: "continuous", canvas: document.getElementById("cont-canvas"), smooth: true },
  ];

  const lut = buildLut(BLUES, 255);

  function buildLut(stops, n) {
    const rgb = stops.map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
    const out = [];
    for (let i = 0; i < n; i++) {
      const t = (i / (n - 1)) * (rgb.length - 1);
      const a = Math.floor(t);
      const b = Math.min(a + 1, rgb.length - 1);
      const f = t - a;
      out.push(rgb[a].map((c, j) => Math.round(c + (rgb[b][j] - c) * f)));
    }
    return out;
  }

  function decode(b64) {
    const s = atob(b64);
    const out = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
    return out;
  }

  // position of cell idx in normalized [0, 1] panel coordinates
  function cellPosition(panelData, idx) {
    const [row, col] = panelData.cells[idx];
    return { x: (col + 0.5) / panelData.res, y: (row + 0.5) / panelData.res };
  }

  function nearestCell(panelData, u, v) {
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < panelData.cells.length; i++) {
      const p = cellPosition(panelData, i);
      const d = (p.x - u) ** 2 + (p.y - v) ** 2;
      if (d < bestDist) { bestDist = d; best = i; }
    }
    return best;
  }

  function selectedCell(panelData) {
    return state.u === null ? panelData.default : nearestCell(panelData, state.u, state.v);
  }

  function resizeCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const size = Math.round(canvas.clientWidth * dpr);
    if (size > 0 && canvas.width !== size) { canvas.width = size; canvas.height = size; }
  }

  function drawStar(ctx, x, y, r) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const radius = i % 2 === 0 ? r : r * 0.45;
      const a = -Math.PI / 2 + (i * Math.PI) / 5;
      ctx.lineTo(x + radius * Math.cos(a), y + radius * Math.sin(a));
    }
    ctx.closePath();
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.lineWidth = Math.max(1, r * 0.12);
    ctx.strokeStyle = "#000000";
    ctx.stroke();
  }

  function renderPanel(panel) {
    const panelData = state.data[panel.key];
    const k = state.data.ks[state.kIndex];
    const idx = selectedCell(panelData);
    const values = decode(panelData.maps[k][state.quantity][idx]);
    const res = panelData.res;

    // draw the map at its native resolution, then upscale onto the visible canvas
    const off = panel.offscreen || (panel.offscreen = document.createElement("canvas"));
    off.width = res;
    off.height = res;
    const offCtx = off.getContext("2d");
    const img = offCtx.createImageData(res, res);
    for (let i = 0; i < values.length; i++) {
      const c = values[i] === WALL ? WALL_RGB : lut[values[i]];
      img.data[4 * i] = c[0];
      img.data[4 * i + 1] = c[1];
      img.data[4 * i + 2] = c[2];
      img.data[4 * i + 3] = 255;
    }
    offCtx.putImageData(img, 0, 0);

    resizeCanvas(panel.canvas);
    const ctx = panel.canvas.getContext("2d");
    const size = panel.canvas.width;
    ctx.imageSmoothingEnabled = panel.smooth;
    ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, size, size);
    ctx.drawImage(off, 0, 0, size, size);

    const p = cellPosition(panelData, idx);
    drawStar(ctx, p.x * size, p.y * size, size * 0.035);
  }

  function render() {
    if (!state.data) return;
    const k = state.data.ks[state.kIndex];
    els.kOut.textContent = String(k);
    els.slider.setAttribute("aria-valuetext", `k = ${k}`);
    els.markerRole.textContent = state.quantity === "q" ? "goal" : "start state";
    panels.forEach(renderPanel);
    els.bars.querySelectorAll(".bar").forEach((bar, i) => bar.classList.toggle("active", i === state.kIndex));
  }

  function buildTicksAndBars() {
    const { ks, test_return: testReturn } = state.data;
    els.slider.max = String(ks.length - 1);
    els.ticks.innerHTML = ks.map((k) => `<span>${k}</span>`).join("");
    els.bars.style.gridTemplateColumns = `repeat(${ks.length}, 1fr)`;
    els.bars.innerHTML = ks
      .map((k, i) => {
        const r = testReturn[k];
        const label = r === undefined ? "–" : `${Math.round(r * 100)}%`;
        const height = r === undefined ? 0 : Math.max(2, r * 120);
        return `<div class="bar" data-index="${i}" title="k = ${k}: ${label}">
          <span class="value">${label}</span>
          <span class="fill" style="height:${height}px"></span>
          <span class="name">k = ${k}</span>
        </div>`;
      })
      .join("");
    els.bars.querySelectorAll(".bar").forEach((bar) =>
      bar.addEventListener("click", () => setK(Number(bar.dataset.index)))
    );
  }

  function setK(i) {
    state.kIndex = i;
    els.slider.value = String(i);
    render();
  }

  function bindEvents() {
    els.slider.addEventListener("input", () => setK(Number(els.slider.value)));

    els.quantityButtons.forEach((btn) =>
      btn.addEventListener("click", () => {
        state.quantity = btn.dataset.quantity;
        els.quantityButtons.forEach((b) => b.setAttribute("aria-checked", String(b === btn)));
        render();
      })
    );

    // click / drag on either map moves the goal (Q) or start (SR) in both maps
    panels.forEach(({ canvas }) => {
      canvas.tabIndex = 0;
      const moveTo = (e) => {
        const rect = canvas.getBoundingClientRect();
        state.u = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
        state.v = Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height));
        render();
      };
      canvas.addEventListener("pointerdown", (e) => { canvas.setPointerCapture(e.pointerId); moveTo(e); });
      canvas.addEventListener("pointermove", (e) => { if (canvas.hasPointerCapture(e.pointerId)) moveTo(e); });
      // arrow keys move the marker by one grid-world cell
      canvas.addEventListener("keydown", (e) => {
        const step = 1 / state.data.discrete.res;
        const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
        if (!moves[e.key]) return;
        e.preventDefault();
        if (state.u === null) {
          const p = cellPosition(state.data.discrete, state.data.discrete.default);
          state.u = p.x;
          state.v = p.y;
        }
        state.u = Math.min(1, Math.max(0, state.u + moves[e.key][0]));
        state.v = Math.min(1, Math.max(0, state.v + moves[e.key][1]));
        render();
      });
    });

    window.addEventListener("resize", render);
  }

  // assets/heatmaps.js (loaded before this script) sets window.HEATMAP_DATA
  if (!window.HEATMAP_DATA) {
    els.status.textContent = "Could not load assets/heatmaps.js. Re-run the Website cells of notebooks/cover_figure.ipynb.";
    return;
  }
  state.data = window.HEATMAP_DATA;
  buildTicksAndBars();
  bindEvents();
  render();
  els.status.textContent = "";
})();
