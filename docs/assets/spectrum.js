// Spectrum widget (FB rank line + share of the SR beyond it): eigenvalues of the discrete four-rooms SR M_k = (I - gamma P^k)^{-1} as a function of k.
// Data comes from notebooks/cover_figure.ipynb ("Website -- spectrum" cells) -> docs/assets/spectrum_data.js
// (window.SPECTRUM_DATA). M_k shares the eigenvectors of P for every k, with eigenvalues
// mu_i(k) = 1 / (1 - gamma lambda_i^k), so mu is computed here and only lambda and the eigenvectors are stored.
(function () {
  "use strict";

  const WALL_RGB = [230, 230, 230];
  // matplotlib "RdBu_r": blue = negative, red = positive (eigenvectors are signed)
  const RDBU = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"];
  const SVG_NS = "http://www.w3.org/2000/svg";
  const MARGIN = { top: 14, right: 14, bottom: 42, left: 46 };
  // the plot shows the largest eigenvalues only: the rest have lambda <= 0.2, down to -0.6 (the checkerboard parity
  // of the grid), and their mu just oscillates around 1 between odd and even k. Rank measures use the full spectrum.
  const SHOWN = 150;

  // d = rank of the FB bottleneck drawn on the plot
  const state = { data: null, k: 1, d: 10, xMode: "index", yMode: "log", pinned: 1, hover: null };
  const els = {
    slider: document.getElementById("spec-k-slider"),
    kOut: document.getElementById("spec-k-out"),
    ticks: document.getElementById("spec-k-ticks"),
    dSlider: document.getElementById("spec-d-slider"),
    dOut: document.getElementById("spec-d-out"),
    xButtons: document.querySelectorAll("#spec-x-mode button"),
    plot: document.getElementById("spec-plot"),
    readout: document.getElementById("spec-readout"),
    vecCanvas: document.getElementById("spec-vec"),
    vecCaption: document.getElementById("spec-vec-caption"),
    thumbs: document.getElementById("spec-thumbs"),
    status: document.getElementById("spec-status"),
  };

  const lut = buildLut(RDBU, 255);

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

  function decodeInt8(b64) {
    const s = atob(b64);
    const out = new Int8Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = (s.charCodeAt(i) << 24) >> 24;
    return out;
  }

  function mu(lambda, k) {
    return 1 / (1 - state.data.gamma * Math.pow(lambda, k));
  }

  // how much of mode i survives at k: 0 when mu = 1 (fully filtered), 1 at the top eigenvalue 1 / (1 - gamma)
  function weight(m) {
    return Math.min(1, Math.abs(Math.log(m)) / Math.log(1 / (1 - state.data.gamma)));
  }

  function fmt(x) {
    return x >= 10 ? x.toFixed(1) : x.toFixed(2);
  }

  // stable rank, normalized spectral entropy and participation ratio of M_k from its singular values (as in the notebook)
  function rankMeasures(sv) {
    let s2 = 0;
    let s4 = 0;
    for (const s of sv) { s2 += s * s; s4 += s ** 4; }
    let entropy = 0;
    for (const s of sv) { const p = (s * s) / s2; entropy -= p * Math.log(p + 1e-20); }
    return { stable: s2 / (sv[0] * sv[0]), entropy: entropy / Math.log(sv.length), participation: (s2 * s2) / s4 };
  }

  // share of M_k - I (the SR without its identity term, which every mode keeps) that lies outside the d largest modes,
  // i.e. what a rank-d FB model cannot represent: sum of (mu_i - 1)^2 beyond the top d over the total
  function tailShare(k, d) {
    const e = state.data.lambda.map((l) => (mu(l, k) - 1) ** 2).sort((a, b) => b - a);
    let total = 0;
    let tail = 0;
    e.forEach((v, i) => { total += v; if (i >= d) tail += v; });
    return tail / total;
  }

  function pct(x) {
    return x >= 0.1 ? `${Math.round(x * 100)}%` : `${(x * 100).toFixed(1)}%`;
  }

  // --- eigenvector maps ---

  function drawVector(canvas, i) {
    const d = state.data;
    const n = d.lambda.length;
    const off = canvas.offscreen || (canvas.offscreen = document.createElement("canvas"));
    off.width = d.cols;
    off.height = d.rows;
    const offCtx = off.getContext("2d");
    const img = offCtx.createImageData(d.cols, d.rows);
    for (let p = 0; p < d.rows * d.cols; p++) {
      img.data[4 * p] = WALL_RGB[0];
      img.data[4 * p + 1] = WALL_RGB[1];
      img.data[4 * p + 2] = WALL_RGB[2];
      img.data[4 * p + 3] = 255;
    }
    for (let c = 0; c < n; c++) {
      const [row, col] = d.cells[c];
      const v = d.vecs[i * n + c] / 127;
      const rgb = lut[Math.round(((v + 1) / 2) * (lut.length - 1))];
      const p = row * d.cols + col;
      img.data[4 * p] = rgb[0];
      img.data[4 * p + 1] = rgb[1];
      img.data[4 * p + 2] = rgb[2];
    }
    offCtx.putImageData(img, 0, 0);

    const dpr = window.devicePixelRatio || 1;
    const size = Math.round(canvas.clientWidth * dpr);
    if (size > 0 && canvas.width !== size) { canvas.width = size; canvas.height = size; }
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }

  function shownIndex() {
    return state.hover === null ? state.pinned : state.hover;
  }

  function renderVector() {
    const i = shownIndex();
    const lambda = state.data.lambda[i];
    drawVector(els.vecCanvas, i);
    els.vecCaption.textContent = `Eigenvector i = ${i + 1}: λ = ${lambda.toFixed(3)}, μ(k = ${state.k}) = ${fmt(mu(lambda, state.k))}`;
  }

  function buildThumbs() {
    // 138 is the last mode before the block of equal lambda = 0.2, whose eigenvectors are not unique
    const groups = [
      { title: "Low frequency", indices: [2, 3, 5, 10] },
      { title: "Higher frequency", indices: [50, 100, 120, 138] },
    ];
    els.thumbs.innerHTML = groups
      .map((g) => `<div class="thumb-group">
        <span class="thumb-title">${g.title}</span>
        <div class="thumb-row">${g.indices
          .map((i) => `<button type="button" class="thumb" data-index="${i - 1}" aria-label="Show eigenvector ${i}">
            <canvas></canvas>
            <span class="thumb-name">i = ${i}</span>
            <span class="thumb-mu"></span>
          </button>`)
          .join("")}</div>
      </div>`)
      .join("");
    els.thumbs.querySelectorAll(".thumb").forEach((t) => {
      const i = Number(t.dataset.index);
      t.addEventListener("click", () => { state.pinned = i; state.hover = null; render(); });
      t.addEventListener("pointerenter", () => { state.hover = i; renderHover(); });
      t.addEventListener("pointerleave", () => { state.hover = null; renderHover(); });
    });
  }

  function renderThumbs() {
    els.thumbs.querySelectorAll(".thumb").forEach((t) => {
      const i = Number(t.dataset.index);
      const m = mu(state.data.lambda[i], state.k);
      t.style.setProperty("--weight", String(0.2 + 0.8 * weight(m)));
      t.querySelector(".thumb-mu").textContent = `μ = ${fmt(m)}`;
      t.classList.toggle("active", i === shownIndex());
      const canvas = t.querySelector("canvas");
      if (!canvas.drawn || canvas.width !== Math.round(canvas.clientWidth * (window.devicePixelRatio || 1))) {
        drawVector(canvas, i);
        canvas.drawn = true;
      }
    });
  }

  // --- spectrum plot (SVG) ---

  function el(name, attrs, parent) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (parent) parent.appendChild(node);
    return node;
  }

  function scales(width, height) {
    const d = state.data;
    const n = d.shown;
    const top = 1 / (1 - d.gamma);
    const x0 = MARGIN.left;
    const x1 = width - MARGIN.right;
    const y0 = height - MARGIN.bottom;
    const y1 = MARGIN.top;
    const lamMin = d.lambda[n - 1];
    const x = state.xMode === "index"
      ? (i) => x0 + ((i - 1) / (n - 1)) * (x1 - x0)
      : (l) => x0 + ((l - lamMin) / (1 - lamMin)) * (x1 - x0);
    const yMin = state.yMode === "log" ? 0.9 : 0;
    const yMax = top * 1.08;
    const y = state.yMode === "log"
      ? (v) => y0 - (Math.log(v / yMin) / Math.log(yMax / yMin)) * (y0 - y1)
      : (v) => y0 - ((v - yMin) / (yMax - yMin)) * (y0 - y1);
    const yTicks = state.yMode === "log" ? [1, 2, 5, 10, 20] : [0, 5, 10, 15, 20];
    const xTicks = state.xMode === "index" ? [1, 25, 50, 75, 100, 125, n] : [0.2, 0.4, 0.6, 0.8, 1];
    return { x, y, x0, x1, y0, y1, yMin, yTicks, xTicks, top, lamMin };
  }

  // x position of eigenvalue i in the current x mode
  function xOf(sc, i) {
    return state.xMode === "index" ? sc.x(i + 1) : sc.x(state.data.lambda[i]);
  }

  function renderPlot() {
    const d = state.data;
    const svg = els.plot;
    const width = Math.max(280, Math.round(svg.getBoundingClientRect().width));
    const height = Math.round(Math.min(360, Math.max(240, width * 0.55)));
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.style.height = `${height}px`;
    svg.innerHTML = "";
    const sc = scales(width, height);
    state.scales = sc;

    // axes and grid
    for (const t of sc.yTicks) {
      el("line", { x1: sc.x0, x2: sc.x1, y1: sc.y(t), y2: sc.y(t), class: "grid" }, svg);
      el("text", { x: sc.x0 - 6, y: sc.y(t), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, svg).textContent = String(t);
    }
    for (const t of sc.xTicks) {
      el("text", { x: sc.x(t), y: sc.y0 + 16, class: "tick", "text-anchor": "middle" }, svg).textContent = String(t);
    }
    el("line", { x1: sc.x0, x2: sc.x1, y1: sc.y0, y2: sc.y0, class: "axis" }, svg);
    el("text", { x: (sc.x0 + sc.x1) / 2, y: sc.y0 + 34, class: "axis-label", "text-anchor": "middle" }, svg).textContent =
      state.xMode === "index" ? "index i (sorted by λᵢ of P, low frequency first)" : "eigenvalue λᵢ of P";
    el("text", { x: 12, y: (sc.y0 + sc.y1) / 2, class: "axis-label", "text-anchor": "middle", transform: `rotate(-90 12 ${(sc.y0 + sc.y1) / 2})` }, svg).textContent = "μᵢ(k)";

    // reference lines: the top eigenvalue 1 / (1 - gamma) and the floor mu = 1 that filtered modes collapse to
    el("line", { x1: sc.x0, x2: sc.x1, y1: sc.y(sc.top), y2: sc.y(sc.top), class: "ref" }, svg);
    el("text", { x: sc.x1 - 4, y: sc.y(sc.top) - 5, class: "ref-label", "text-anchor": "end" }, svg).textContent = "1/(1−γ)";
    el("line", { x1: sc.x0, x2: sc.x1, y1: sc.y(1), y2: sc.y(1), class: "ref" }, svg);
    el("text", { x: sc.x1 - 4, y: sc.y(1) - 5, class: "ref-label", "text-anchor": "end" }, svg).textContent = "μ = 1 (mode filtered out)";

    if (state.xMode === "lambda") {
      // the filter curve mu(lambda) = 1 / (1 - gamma lambda^k), for k = 1 (ghost) and the current k
      for (const [k, cls] of [[1, "ghost-curve"], [state.k, "curve"]]) {
        const pts = [];
        for (let s = 0; s <= 300; s++) {
          const l = sc.lamMin + (s / 300) * (1 - sc.lamMin);
          pts.push(`${sc.x(l).toFixed(1)},${sc.y(mu(l, k)).toFixed(1)}`);
        }
        el("polyline", { points: pts.join(" "), class: cls }, svg);
      }
    } else {
      // FB rank budget: modes right of the line are what a rank-d model has to drop
      const xd = sc.x(state.d + 0.5);
      el("rect", { x: xd, y: sc.y1, width: Math.max(0, sc.x1 - xd), height: sc.y0 - sc.y1, class: "cut-region" }, svg);
      el("line", { x1: xd, x2: xd, y1: sc.y1, y2: sc.y0, class: "rank-line" }, svg);
      el("text", { x: xd + 5, y: sc.y1 + 30, class: "rank-label" }, svg).textContent = `FB rank d = ${state.d}`;
      el("text", { x: xd + 5, y: sc.y1 + 45, class: "rank-label muted" }, svg).textContent = "beyond: FB can't hold";

      const ghost = el("g", { class: "ghost" }, svg);
      d.lambda.slice(0, d.shown).forEach((l, i) => el("circle", { cx: xOf(sc, i), cy: sc.y(mu(l, 1)), r: 1.8 }, ghost));
    }

    const points = el("g", { class: "points" }, svg);
    d.lambda.slice(0, d.shown).forEach((l, i) => el("circle", { cx: xOf(sc, i), cy: sc.y(mu(l, state.k)), r: 2.6 }, points));

    // markers for the thumbnail modes, and the shown eigenvector
    const marks = el("g", { class: "marks" }, svg);
    els.thumbs.querySelectorAll(".thumb").forEach((t) => {
      const i = Number(t.dataset.index);
      el("circle", { cx: xOf(sc, i), cy: sc.y(mu(d.lambda[i], state.k)), r: 5 }, marks);
    });
    state.highlight = el("circle", { r: 7.5, class: "highlight" }, svg);
    positionHighlight();

    const share = tailShare(state.k, state.d);
    const share1 = tailShare(1, state.d);
    const stable = rankMeasures(d.sv[state.k]).stable;
    const stable1 = rankMeasures(d.sv[1]).stable;
    const was = (v) => (state.k === 1 ? "" : ` <span class="muted">(k = 1: ${v})</span>`);
    els.readout.innerHTML = `SR beyond rank ${state.d}: <b>${pct(share)}</b>${was(pct(share1))}` +
      ` · stable rank <b>${stable.toFixed(1)}</b>${was(stable1.toFixed(1))}`;
  }

  function positionHighlight() {
    const i = shownIndex();
    const sc = state.scales;
    state.highlight.setAttribute("cx", xOf(sc, i));
    state.highlight.setAttribute("cy", sc.y(mu(state.data.lambda[i], state.k)));
  }

  function nearestIndex(e) {
    const rect = els.plot.getBoundingClientRect();
    const vb = els.plot.viewBox.baseVal;
    const px = ((e.clientX - rect.left) / rect.width) * vb.width;
    const py = ((e.clientY - rect.top) / rect.height) * vb.height;
    const sc = state.scales;
    let best = 0;
    let bestDist = Infinity;
    state.data.lambda.slice(0, state.data.shown).forEach((l, i) => {
      const dist = (xOf(sc, i) - px) ** 2 + (sc.y(mu(l, state.k)) - py) ** 2;
      if (dist < bestDist) { bestDist = dist; best = i; }
    });
    return best;
  }

  // --- rendering ---

  function renderHover() {
    positionHighlight();
    renderVector();
    els.thumbs.querySelectorAll(".thumb").forEach((t) => t.classList.toggle("active", Number(t.dataset.index) === shownIndex()));
  }

  function render() {
    if (!state.data) return;
    els.kOut.textContent = String(state.k);
    els.slider.setAttribute("aria-valuetext", `k = ${state.k}`);
    renderPlot();
    renderThumbs();
    renderVector();
  }

  function bindSegmented(buttons, key) {
    buttons.forEach((btn) =>
      btn.addEventListener("click", () => {
        state[key] = btn.dataset.value;
        buttons.forEach((b) => b.setAttribute("aria-checked", String(b === btn)));
        render();
      })
    );
  }

  function bindEvents() {
    els.slider.addEventListener("input", () => { state.k = Number(els.slider.value); render(); });
    els.dSlider.addEventListener("input", () => { state.d = Number(els.dSlider.value); els.dOut.textContent = String(state.d); render(); });
    bindSegmented(els.xButtons, "xMode");

    // hover shows an eigenvector, click / tap pins it
    els.plot.addEventListener("pointermove", (e) => { state.hover = nearestIndex(e); renderHover(); });
    els.plot.addEventListener("pointerleave", () => { state.hover = null; renderHover(); });
    els.plot.addEventListener("pointerdown", (e) => { state.pinned = nearestIndex(e); state.hover = null; renderHover(); });
    // arrow keys step through the eigenvalues
    els.plot.addEventListener("keydown", (e) => {
      const step = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
      if (!step) return;
      e.preventDefault();
      state.pinned = Math.min(state.data.shown - 1, Math.max(0, state.pinned + step));
      state.hover = null;
      renderHover();
    });

    window.addEventListener("resize", render);
  }

  // assets/spectrum_data.js (loaded before this script) sets window.SPECTRUM_DATA
  if (!window.SPECTRUM_DATA) {
    els.status.textContent = "Could not load assets/spectrum_data.js. Re-run the spectrum cells of notebooks/cover_figure.ipynb.";
    return;
  }
  state.data = { ...window.SPECTRUM_DATA, vecs: decodeInt8(window.SPECTRUM_DATA.vecs) };
  state.data.shown = Math.min(SHOWN, state.data.lambda.length);
  els.slider.max = String(state.data.k_max);
  els.ticks.innerHTML = [1, 10, 20, 30, 40].filter((k) => k <= state.data.k_max).map((k) => `<span>${k}</span>`).join("");
  buildThumbs();
  bindEvents();
  render();
  els.status.textContent = "";
})();
