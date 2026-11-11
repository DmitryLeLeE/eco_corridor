/* EcoCorridor — canvas map engine (no external map libraries) */

class MapView {
  constructor(canvas, tooltipEl) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.tooltip = tooltipEl;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = 1;
    this.h = 1;
    this.bounds = null;
    this.view = { cx: 0, cy: 0, scale: 0.01 };
    this.rasters = {
      hillshade: { img: null, visible: true, alpha: 1, z: 0 },
      landcover: { img: null, visible: false, alpha: 1, z: 1 },
      resistance: { img: null, visible: true, alpha: 0.93, z: 2 },
      cost: { img: null, visible: false, alpha: 0.85, z: 3 },
    };
    this.data = { cores: [], corridors: [], corridorArea: [], pinchpoints: [], roads: [], rivers: [], settlements: [] };
    this.showRoads = true;
    this.showLabels = true;
    this.hoverHit = null;
    this.highlight = null;
    this.route = null;
    this.routePoints = [];
    this.rasterDirty = true;
    this.offscreen = document.createElement("canvas");
    this.offctx = this.offscreen.getContext("2d");
    this._dragging = false;
    this._moved = false;
    this._last = null;
    this._onHover = null;
    this._onPick = null;
    this._onView = null;

    this._bindEvents();
    this.resize();
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  set onHover(fn) { this._onHover = fn; }
  set onPick(fn) { this._onPick = fn; }
  set onView(fn) { this._onView = fn; }

  /* ------------------------------------------------------------ geometry */
  setProject(bounds) {
    this.bounds = bounds;
    this.fit();
  }

  fit(padding = 0.965) {
    if (!this.bounds) return;
    const [l, b, r, t] = this.bounds;
    this.view.cx = (l + r) / 2;
    this.view.cy = (b + t) / 2;
    const sx = (this.w * padding) / (r - l);
    const sy = (this.h * padding) / (t - b);
    this.view.scale = Math.min(sx, sy);
    this.minScale = this.view.scale * 0.92;
    this.maxScale = 0.5;
    this.rasterDirty = true;
    this._emitView();
  }

  sx(x) { return (x - this.view.cx) * this.view.scale + this.w / 2; }
  sy(y) { return (this.view.cy - y) * this.view.scale + this.h / 2; }
  wx(px) { return this.view.cx + (px - this.w / 2) / this.view.scale; }
  wy(py) { return this.view.cy - (py - this.h / 2) / this.view.scale; }

  toScreen([x, y]) { return [this.sx(x), this.sy(y)]; }
  toWorld([px, py]) { return [this.wx(px), this.wy(py)]; }

  zoomAt(factor, px, py) {
    const wx = this.wx(px);
    const wy = this.wy(py);
    const next = Math.min(this.maxScale, Math.max(this.minScale, this.view.scale * factor));
    if (next === this.view.scale) return;
    this.view.scale = next;
    this.view.cx = wx - (px - this.w / 2) / next;
    this.view.cy = wy + (py - this.h / 2) / next;
    this._clampCentre();
    this.rasterDirty = true;
    this._emitView();
  }

  zoomBy(factor) { this.zoomAt(factor, this.w / 2, this.h / 2); }

  focusBounds([l, b, r, t], padding = 0.75) {
    if (!this.bounds) return;
    this.view.cx = (l + r) / 2;
    this.view.cy = (b + t) / 2;
    const s = Math.min((this.w * padding) / Math.max(r - l, 1), (this.h * padding) / Math.max(t - b, 1));
    this.view.scale = Math.min(this.maxScale, Math.max(this.minScale, s));
    this.rasterDirty = true;
    this._emitView();
  }

  _clampCentre() {
    const [l, b, r, t] = this.bounds;
    const halfW = this.w / (2 * this.view.scale);
    const halfH = this.h / (2 * this.view.scale);
    this.view.cx = Math.min(r, Math.max(l, this.view.cx));
    this.view.cy = Math.min(t, Math.max(b, this.view.cy));
    if (r - l < halfW * 2) this.view.cx = (l + r) / 2;
    if (t - b < halfH * 2) this.view.cy = (b + t) / 2;
  }

  _emitView() { if (this._onView) this._onView(this.view); }

  /* --------------------------------------------------------------- data */
  clearImages() {
    for (const raster of Object.values(this.rasters)) raster.img = null;
    this.highlight = null;
    this.hoverHit = null;
    this.rasterDirty = true;
  }

  setImages(layers) {
    for (const [name, url] of Object.entries(layers)) {
      if (!this.rasters[name] || !url) continue;
      const img = new Image();
      img.decoding = "async";
      img.onload = () => {
        this.rasters[name].img = img;
        this.rasterDirty = true;
      };
      img.src = url + (url.includes("?") ? "&" : "?") + "v=" + Date.now();
    }
  }

  setData(project, analysis) {
    const toFeatures = (fc, kind, prefix) => (fc?.features || []).map((f, i) => ({ ...f.properties, geometry: f.geometry, kind, _id: `${prefix}${f.properties?.id ?? i}` }));
    this.data.cores = (analysis?.cores || project.layers.cores)?.features.map((f, i) => ({
      ...f.properties, geometry: f.geometry, kind: "core", _id: "c" + (f.properties?.core_id ?? i),
    })) || [];
    this.data.corridors = toFeatures(analysis?.corridors, "corridor", "l");
    this.data.corridorArea = toFeatures(analysis?.corridor_area, "corridorArea", "l");
    this.data.pinchpoints = (analysis?.pinchpoints?.features || []).map((f, i) => ({ ...f.properties, geometry: f.geometry, kind: "pinch", _id: "p" + i }));
    this.data.roads = toFeatures(project.layers.roads, "road", "r");
    this.data.rivers = toFeatures(project.layers.rivers, "river", "v");
    this.data.settlements = toFeatures(project.layers.settlements, "settlement", "s");
    this._indexCores();
    this.rasterDirty = true;
  }

  _indexCores() {
    this.coreIndex = this.data.cores.map((core) => {
      const rings = [];
      const g = core.geometry;
      if (!g) return { core, rings };
      const push = (poly) => rings.push(poly[0]);
      if (g.type === "Polygon") push(g.coordinates);
      if (g.type === "MultiPolygon") g.coordinates.forEach(push);
      return { core, rings, bbox: this._bbox(g) };
    });
  }

  _bbox(geometry) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    const walk = (coords) => {
      if (typeof coords[0] === "number") {
        minX = Math.min(minX, coords[0]); maxX = Math.max(maxX, coords[0]);
        minY = Math.min(minY, coords[1]); maxY = Math.max(maxY, coords[1]);
      } else coords.forEach(walk);
    };
    walk(geometry.coordinates);
    return [minX, minY, maxX, maxY];
  }

  setLayer(name, visible) {
    if (!this.rasters[name]) return;
    this.rasters[name].visible = visible;
    this.rasterDirty = true;
  }

  setHighlight(hit) { this.highlight = hit; }

  setRoute(route) {
    this.route = route;
    this.routePoints = route?.points || [];
  }

  clearRoute() { this.route = null; this.routePoints = []; }

  /* ------------------------------------------------------------- events */
  _bindEvents() {
    const canvas = this.canvas;

    canvas.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      canvas.setPointerCapture(event.pointerId);
      this._dragging = true;
      this._moved = false;
      this._last = [event.offsetX, event.offsetY];
      canvas.classList.add("dragging");
    });

    canvas.addEventListener("pointermove", (event) => {
      const px = event.offsetX;
      const py = event.offsetY;
      if (this._dragging && this._last) {
        const dx = px - this._last[0];
        const dy = py - this._last[1];
        if (Math.abs(dx) + Math.abs(dy) > 2) this._moved = true;
        this.view.cx -= dx / this.view.scale;
        this.view.cy += dy / this.view.scale;
        this._clampCentre();
        this._last = [px, py];
        this.rasterDirty = true;
        this._emitView();
      }
      this._handleHover(px, py, event);
    });

    const end = (event) => {
      if (!this._dragging) return;
      this._dragging = false;
      canvas.classList.remove("dragging");
      this._last = null;
      if (!this._moved && this._onPick) {
        this._onPick(this.toWorld([event.offsetX, event.offsetY]));
      }
    };
    canvas.addEventListener("pointerup", end);
    canvas.addEventListener("pointercancel", () => { this._dragging = false; canvas.classList.remove("dragging"); });

    canvas.addEventListener("pointerleave", () => {
      this.hoverHit = null;
      if (this._onHover) this._onHover(null);
    });

    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const factor = Math.exp(-event.deltaY * 0.0016);
      this.zoomAt(factor, event.offsetX, event.offsetY);
    }, { passive: false });

    canvas.addEventListener("dblclick", (event) => {
      event.preventDefault();
      this.zoomAt(1.9, event.offsetX, event.offsetY);
    });

    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.w = Math.max(1, Math.floor(rect.width));
    this.h = Math.max(1, Math.floor(rect.height));
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.floor(this.w * this.dpr);
    this.canvas.height = Math.floor(this.h * this.dpr);
    this.canvas.style.width = this.w + "px";
    this.canvas.style.height = this.h + "px";
    this.offscreen.width = this.canvas.width;
    this.offscreen.height = this.canvas.height;
    this.rasterDirty = true;
    if (this.bounds && !this._sized) {
      this._sized = true;
      this.fit();
    }
  }

  _handleHover(px, py, event) {
    const hit = this._hitTest(px, py);
    const changed = JSON.stringify(hit?.id) !== JSON.stringify(this.hoverHit?.id) ||
      hit?.type !== this.hoverHit?.type;
    this.hoverHit = hit;
    if (this._onHover) this._onHover(hit, [px, py], this.toWorld([px, py]));
    this.canvas.style.cursor = hit ? "pointer" : (this._dragging ? "grabbing" : "grab");
  }

  _hitTest(px, py) {
    // pinch points first
    for (const pinch of this.data.pinchpoints) {
      const [x, y] = this.toScreen(pinch.geometry.coordinates);
      if (Math.hypot(x - px, y - py) < 11) return { type: "pinch", id: pinch._id, item: pinch };
    }
    // cores
    const world = this.toWorld([px, py]);
    for (const entry of this.coreIndex || []) {
      if (this._pointInRings(world, entry.rings)) return { type: "core", id: entry.core._id, item: entry.core };
    }
    // corridors
    let best = null;
    for (const corridor of this.data.corridors) {
      const dist = this._distanceToLine(px, py, corridor.geometry.coordinates);
      if (dist < 8 && (!best || dist < best.dist)) best = { type: "corridor", dist, item: corridor };
    }
    if (best) return { type: "corridor", id: best.item._id, item: best.item };
    return null;
  }

  _pointInRings(point, rings) {
    let inside = false;
    for (const ring of rings) {
      for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const xi = ring[i][0], yi = ring[i][1];
        const xj = ring[j][0], yj = ring[j][1];
        if ((yi > point[1]) !== (yj > point[1]) &&
            point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi + 1e-12) + xi) inside = !inside;
      }
    }
    return inside;
  }

  _distanceToLine(px, py, coords) {
    let best = Infinity;
    for (let i = 1; i < coords.length; i++) {
      const [x1, y1] = this.toScreen(coords[i - 1]);
      const [x2, y2] = this.toScreen(coords[i]);
      const dx = x2 - x1, dy = y2 - y1;
      const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy || 1)));
      best = Math.min(best, Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)));
    }
    return best;
  }

  /* -------------------------------------------------------------- render */
  _loop() {
    this._draw();
    requestAnimationFrame(this._loop);
  }

  _draw() {
    const ctx = this.ctx;
    const t = performance.now();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    ctx.fillStyle = "#eef1f3";
    ctx.fillRect(0, 0, this.w, this.h);
    if (!this.bounds) return;

    if (this.rasterDirty) this._renderRasters();
    ctx.drawImage(this.offscreen, 0, 0, this.w, this.h);

    const [l, b, r, tp] = this.bounds;
    const x0 = this.sx(l), y0 = this.sy(tp), x1 = this.sx(r), y1 = this.sy(b);
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0, y0, x1 - x0, y1 - y0);
    ctx.clip();

    this._drawGraticule(ctx, x0, y0, x1, y1);
    if (this.showRoads) this._drawHydro(ctx);
    this._drawCorridorAreas(ctx);
    this._drawCores(ctx);
    this._drawCorridors(ctx, t);
    this._drawPinchpoints(ctx, t);
    if (this.showRoads) this._drawSettlements(ctx);
    if (this.showLabels) this._drawLabels(ctx, x0, y0);
    this._drawRoute(ctx, t);
    ctx.restore();
  }

  _renderRasters() {
    const ctx = this.offctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    const [l, b, r, t] = this.bounds;
    const x0 = this.sx(l), y0 = this.sy(t);
    const w = this.sx(r) - x0;
    const h = this.sy(b) - y0;
    const order = Object.values(this.rasters).filter((layer) => layer.visible && layer.img).sort((a, z) => a.z - z.z);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    for (const layer of order) {
      ctx.globalAlpha = layer.alpha;
      ctx.drawImage(layer.img, x0, y0, w, h);
    }
    ctx.globalAlpha = 1;
    this.rasterDirty = false;
  }

  _drawGraticule(ctx, x0, y0, x1, y1) {
    if (!this.bounds) return;
    const step = this._niceStep(110 / this.view.scale);
    const [l, , , t] = this.bounds;
    const startX = Math.ceil(l / step) * step;
    const startY = Math.ceil((this.bounds[1]) / step) * step;
    ctx.save();
    ctx.strokeStyle = "rgba(40, 60, 80, 0.13)";
    ctx.lineWidth = 1;
    ctx.font = "9px 'Cascadia Mono', monospace";
    ctx.fillStyle = "rgba(40, 60, 80, 0.62)";
    ctx.beginPath();
    for (let x = startX; x < this.bounds[2]; x += step) {
      const px = Math.round(this.sx(x)) + 0.5;
      if (px < x0 || px > x1) continue;
      ctx.moveTo(px, y0); ctx.lineTo(px, y1);
      ctx.fillText((x / 1000).toFixed(0) + " км", px + 4, y0 + 12);
    }
    for (let y = startY; y < t; y += step) {
      const py = Math.round(this.sy(y)) + 0.5;
      if (py < y0 || py > y1) continue;
      ctx.moveTo(x0, py); ctx.lineTo(x1, py);
      ctx.fillText((y / 1000).toFixed(0), x0 + 4, py - 4);
    }
    ctx.stroke();
    ctx.restore();
  }

  _niceStep(raw) {
    const pow = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / pow;
    const mult = norm < 1.5 ? 1 : norm < 3.5 ? 2 : norm < 7.5 ? 5 : 10;
    return mult * pow;
  }

  _drawHydro(ctx) {
    ctx.save();
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(30, 110, 190, 0.75)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const river of this.data.rivers) {
      this._traceGeometry(ctx, river.geometry, 0);
    }
    ctx.stroke();
    ctx.strokeStyle = "rgba(55, 55, 60, 0.6)";
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    for (const road of this.data.roads) this._traceGeometry(ctx, road.geometry, 0);
    ctx.stroke();
    ctx.restore();
  }

  _traceGeometry(ctx, geometry, offset) {
    if (!geometry) return;
    if (geometry.type === "LineString") {
      geometry.coordinates.forEach((coord, i) => {
        const [x, y] = this.toScreen(coord);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
    } else if (geometry.type === "MultiLineString" || geometry.type === "Polygon") {
      const list = geometry.type === "Polygon" ? geometry.coordinates : geometry.coordinates;
      list.forEach((part) => this._traceGeometry(ctx, { type: "LineString", coordinates: part }, offset));
    } else if (geometry.type === "MultiPolygon") {
      geometry.coordinates.forEach((poly) => poly.forEach((part) => this._traceGeometry(ctx, { type: "LineString", coordinates: part }, offset)));
    }
  }

  _drawCorridorAreas(ctx) {
    if (!this.data.corridorArea.length) return;
    const highlighted = this.highlight?.id || null;
    ctx.save();
    for (const area of this.data.corridorArea) {
      const isHi = highlighted === area._id;
      ctx.beginPath();
      this._traceGeometry(ctx, area.geometry, 0);
      ctx.closePath();
      ctx.fillStyle = isHi ? "rgba(15, 90, 160, 0.16)" : "rgba(15, 90, 160, 0.06)";
      ctx.fill();
    }
    ctx.restore();
  }

  _drawCores(ctx) {
    const highlighted = this.highlight?.id || null;
    ctx.save();
    for (const core of this.data.cores) {
      const isHi = highlighted === core._id;
      ctx.beginPath();
      const g = core.geometry;
      if (g.type === "Polygon") this._polygonPath(ctx, g.coordinates);
      if (g.type === "MultiPolygon") g.coordinates.forEach((poly) => this._polygonPath(ctx, poly));
      ctx.fillStyle = isHi ? "rgba(20, 110, 70, 0.28)" : "rgba(20, 110, 70, 0.12)";
      ctx.fill();
      ctx.strokeStyle = isHi ? "#0b4f33" : "#14703f";
      ctx.lineWidth = isHi ? 2.4 : 1.4;
      ctx.setLineDash(isHi ? [] : [6, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();
  }

  _polygonPath(ctx, rings) {
    rings.forEach((ring) => {
      ring.forEach((coord, i) => {
        const [x, y] = this.toScreen(coord);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.closePath();
    });
  }

  _drawCorridors(ctx, t) {
    const highlighted = this.highlight?.id || null;
    const corridors = [...this.data.corridors].sort((a, b) => {
      const rankA = a._id === highlighted ? 3 : a.type === "backbone" ? 2 : 1;
      const rankB = b._id === highlighted ? 3 : b.type === "backbone" ? 2 : 1;
      return rankA - rankB;
    });
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const corridor of corridors) {
      const isHi = highlighted === corridor._id;
      const dim = highlighted && !isHi;
      const color = corridor.type === "backbone" ? "#0a3d91" : "#7a3b00";
      const points = corridor.geometry.coordinates.map((c) => this.toScreen(c));

      ctx.globalAlpha = dim ? 0.26 : 1;
      ctx.beginPath();
      this._smoothPath(ctx, points);
      ctx.strokeStyle = isHi ? "#ffffff" : "rgba(255, 255, 255, 0.92)";
      ctx.lineWidth = isHi ? 7.5 : 5.6;
      ctx.stroke();

      ctx.beginPath();
      this._smoothPath(ctx, points);
      ctx.strokeStyle = color;
      ctx.lineWidth = isHi ? 3.6 : 2.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    ctx.restore();
  }

  _smoothPath(ctx, points) {
    if (!points.length) return;
    ctx.moveTo(points[0][0], points[0][1]);
    if (points.length === 2) { ctx.lineTo(points[1][0], points[1][1]); return; }
    for (let i = 1; i < points.length - 1; i++) {
      const [x1, y1] = points[i];
      const [x2, y2] = points[i + 1];
      ctx.quadraticCurveTo(x1, y1, (x1 + x2) / 2, (y1 + y2) / 2);
    }
    const [lx, ly] = points[points.length - 1];
    ctx.lineTo(lx, ly);
  }

  _drawPinchpoints(ctx, t) {
    const pulse = 0.5 + 0.5 * Math.sin(t / 520);
    ctx.save();
    for (const pinch of this.data.pinchpoints) {
      const [x, y] = this.toScreen(pinch.geometry.coordinates);
      const r = 5 + pulse * 1.6;
      const hi = this.highlight?.id === pinch._id;
      ctx.beginPath();
      ctx.arc(x, y, r + 5 + pulse * 4, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(200, 40, 30, ${0.06 + pulse * 0.05})`;
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(x, y - r);
      ctx.lineTo(x + r, y);
      ctx.lineTo(x, y + r);
      ctx.lineTo(x - r, y);
      ctx.closePath();
      ctx.fillStyle = hi ? "#ff8a80" : "#c62828";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.4;
      ctx.stroke();
    }
    ctx.restore();
  }

  _drawSettlements(ctx) {
    if (this.view.scale < 0.012) return;
    ctx.save();
    ctx.font = "10px 'Segoe UI', sans-serif";
    for (const settlement of this.data.settlements) {
      const [x, y] = this.toScreen(settlement.geometry.coordinates);
      const big = (settlement.population || 0) >= 5000;
      ctx.beginPath();
      ctx.arc(x, y, big ? 3.4 : 2.1, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.strokeStyle = "rgba(20, 30, 40, 0.9)";
      ctx.lineWidth = 1;
      ctx.stroke();
      if (big && this.view.scale > 0.02) {
        ctx.fillStyle = "#1b2733";
        ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
        ctx.lineWidth = 3;
        ctx.strokeText(settlement.name, x + 7, y + 3);
        ctx.fillText(settlement.name, x + 7, y + 3);
      }
    }
    ctx.restore();
  }

  _drawLabels(ctx, x0, y0) {
    if (this.view.scale < 0.018) return;
    ctx.save();
    ctx.font = "600 11px 'Segoe UI', sans-serif";
    ctx.textAlign = "center";
    for (const core of this.data.cores) {
      const rings = [];
      const g = core.geometry;
      if (g.type === "Polygon") rings.push(g.coordinates[0]);
      if (g.type === "MultiPolygon") g.coordinates.forEach((p) => rings.push(p[0]));
      let sx = 0, sy = 0, n = 0;
      rings.forEach((ring) => ring.forEach((c) => { sx += c[0]; sy += c[1]; n += 1; }));
      if (!n) continue;
      const [px, py] = this.toScreen([sx / n, sy / n]);
      if (px < x0 || py < y0) continue;
      const label = core.name || "";
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
      ctx.strokeText(label, px, py);
      ctx.fillStyle = "#0b3d24";
      ctx.fillText(label, px, py);
    }
    ctx.restore();
  }

  _drawRoute(ctx, t) {
    if (!this.route && !this.routePoints.length) return;
    ctx.save();
    if (this.route) {
      const points = this.route.coords.map((c) => this.toScreen(c));
      ctx.beginPath();
      this._smoothPath(ctx, points);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
      ctx.lineWidth = 8;
      ctx.stroke();
      ctx.beginPath();
      this._smoothPath(ctx, points);
      ctx.setLineDash([7, 6]);
      ctx.lineDashOffset = -((t / 45) % 13);
      ctx.strokeStyle = "#b71c1c";
      ctx.lineWidth = 2.6;
      ctx.stroke();
      ctx.setLineDash([]);
      const start = this.toScreen(this.route.coords[0]);
      const end = this.toScreen(this.route.coords[this.route.coords.length - 1]);
      this._marker(ctx, start, "#14703f");
      this._marker(ctx, end, "#b71c1c");
    }
    for (const point of this.routePoints) this._marker(ctx, this.toScreen(point), "#b71c1c");
    ctx.restore();
  }

  _marker(ctx, [x, y], color) {
    ctx.beginPath();
    ctx.arc(x, y, 6.5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.4;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, y, 2, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
}
