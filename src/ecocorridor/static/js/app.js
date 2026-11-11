/* EcoCorridor — application controller */

const App = (() => {
  const $ = (id) => document.getElementById(id);

  const state = {
    project: null,
    params: null,
    scenario: "base",
    result: null,
    compareRows: null,
    layers: { resistance: true, landcover: false, cost: false, hillshade: true, roads: true, labels: true },
    routeMode: false,
    routeStart: null,
    busy: false,
    requestSeq: 0,
    sampleTimer: null,
  };

  let map = null;
  let runTimer = null;

  /* ------------------------------------------------------------- helpers */
  const shortName = (name) => {
    const match = /«(.+?)»/.exec(name || "");
    return match ? match[1] : (name || "—");
  };

  const fmt = (value, digits = 0) => Charts.nf(value, digits);

  function plural(n, one, few, many) {
    const mod10 = n % 10;
    const mod100 = n % 100;
    if (mod10 === 1 && mod100 !== 11) return one;
    if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
    return many;
  }

  function toast(message, isError = false) {
    const node = $("toast");
    node.textContent = message;
    node.classList.toggle("error", isError);
    node.classList.add("visible");
    clearTimeout(node._timer);
    node._timer = setTimeout(() => node.classList.remove("visible"), 4200);
  }

  function setLoading(visible, text) {
    $("loading").classList.toggle("visible", visible);
    if (text) $("loading-text").textContent = text;
    $("status-dot").classList.toggle("busy", visible);
  }

  function setStatus(text) {
    $("status-text").textContent = text;
  }

  /* -------------------------------------------------------------- startup */
  async function init() {
    try {
      const project = await API.project();
      state.project = project;
      state.params = { ...project.defaults };

      $("project-name").textContent = project.name;
      $("project-meta").textContent = `${fmt((project.bounds[2] - project.bounds[0]) / 1000)} × ${fmt((project.bounds[3] - project.bounds[1]) / 1000)} км · ${project.resolution_m} м · ${project.crs}${project.synthetic ? " · синтетический" : ""}`;
      $("status-crs").textContent = project.crs;
      $("status-res").textContent = `${project.resolution_m} м`;

      map = new MapView($("map"), $("tooltip"));
      window.mapView = map;
      window.appState = state;
      map.setProject(project.bounds);
      map.setData(project, null);
      map.showRoads = state.layers.roads;
      map.showLabels = state.layers.labels;
      bindMap();

      buildScenarios(project.scenarios);
      syncControls();
      bindControls();
      bindShortcuts();
      buildLegend("resistance");

      setStatus("Расчёт базового сценария…");
      await runAnalysis("base");
    } catch (error) {
      console.error(error);
      setLoading(false);
      setStatus("Ошибка: " + error.message);
      toast("Не удалось загрузить проект: " + error.message, true);
    }
  }

  /* ----------------------------------------------------------- scenarios */
  function buildScenarios(scenarios) {
    const container = $("scenario-list");
    container.innerHTML = scenarios
      .map(
        (scenario, index) => `
      <button class="scenario" data-id="${scenario.id}" style="--sc-accent:${scenario.accent}">
        <span class="bar"></span>
        <span>
          <span class="nm">${scenario.name}</span>
        </span>
        <span class="idx">${String(index + 1).padStart(2, "0")}</span>
      </button>`
      )
      .join("");
    container.querySelectorAll(".scenario").forEach((node) => {
      node.addEventListener("click", () => selectScenario(node.dataset.id));
    });
  }

  function selectScenario(id) {
    state.scenario = id;
    document.querySelectorAll(".scenario").forEach((node) => node.classList.toggle("active", node.dataset.id === id));
    const scenario = state.project.scenarios.find((s) => s.id === id);
    $("scenario-tag").textContent = scenario?.id || id;
    $("scenario-note").textContent = scenario?.description || "";
    runAnalysis(id);
  }

  /* ------------------------------------------------------------- controls */
  function syncControls() {
    const p = state.params;
    $("w-slope").value = p.slope_weight;
    $("w-road").value = p.road_weight;
    $("w-urban").value = p.urban_weight;
    $("max-link").value = p.max_link_km;
    $("width").value = p.corridor_width_m;
    $("detour").value = p.detour_ratio;
    $("dispersal").value = p.dispersal_m;
    $("mode").value = p.mode;
    updateSliderOutputs();
    document.querySelectorAll(".scenario").forEach((node) => node.classList.toggle("active", node.dataset.id === state.scenario));
    const scenario = state.project?.scenarios.find((s) => s.id === state.scenario);
    if (scenario) {
      $("scenario-tag").textContent = scenario.id;
      $("scenario-note").textContent = scenario.description;
    }
  }

  function updateSliderOutputs() {
    const values = [
      ["w-slope", "w-slope-val", Number($("w-slope").value), (v) => v.toFixed(2)],
      ["w-road", "w-road-val", Number($("w-road").value), (v) => v.toFixed(2)],
      ["w-urban", "w-urban-val", Number($("w-urban").value), (v) => v.toFixed(2)],
      ["max-link", "max-link-val", Number($("max-link").value), (v) => `${v} км`],
      ["width", "width-val", Number($("width").value), (v) => `${v} м`],
      ["detour", "detour-val", Number($("detour").value), (v) => `×${v.toFixed(2)}`],
      ["dispersal", "dispersal-val", Number($("dispersal").value), (v) => `${v / 1000} км`],
    ];
    values.forEach(([inputId, outputId, value, format]) => {
      $(outputId).textContent = format(value);
      const input = $(inputId);
      const min = Number(input.min);
      const max = Number(input.max);
      input.style.setProperty("--fill", `${((value - min) / (max - min)) * 100}%`);
    });
  }

  function readParams() {
    return {
      mode: $("mode").value,
      slope_weight: Number($("w-slope").value),
      road_weight: Number($("w-road").value),
      urban_weight: Number($("w-urban").value),
      max_link_km: Number($("max-link").value),
      corridor_width_m: Number($("width").value),
      detour_ratio: Number($("detour").value),
      dispersal_m: Number($("dispersal").value),
      min_patch_ha: 5,
      road_decay_m: 350,
      urban_decay_m: 1200,
      slope_ref_deg: 22,
    };
  }

  function bindControls() {
    const sliders = ["w-slope", "w-road", "w-urban", "max-link", "width", "detour", "dispersal"];
    sliders.forEach((id) => {
      $(id).addEventListener("input", () => {
        updateSliderOutputs();
        scheduleRun();
      });
    });
    $("mode").addEventListener("change", scheduleRun);
    $("btn-run").addEventListener("click", () => runAnalysis(state.scenario));

    document.querySelectorAll("[data-reset]").forEach((button) => {
      button.addEventListener("click", () => {
        const defaults = state.project.defaults;
        if (button.dataset.reset === "resistance") {
          $("w-slope").value = defaults.slope_weight;
          $("w-road").value = defaults.road_weight;
          $("w-urban").value = defaults.urban_weight;
          $("mode").value = defaults.mode;
        } else {
          $("max-link").value = defaults.max_link_km;
          $("width").value = defaults.corridor_width_m;
          $("detour").value = defaults.detour_ratio;
          $("dispersal").value = defaults.dispersal_m;
        }
        updateSliderOutputs();
        runAnalysis(state.scenario);
      });
    });

    document.querySelectorAll("#layer-switches input[data-layer]").forEach((input) => {
      input.addEventListener("change", () => {
        const name = input.dataset.layer;
        state.layers[name] = input.checked;
        if (["roads", "labels"].includes(name)) {
          map.showRoads = state.layers.roads;
          map.showLabels = state.layers.labels;
        } else {
          map.setLayer(name, input.checked);
          if (["resistance", "landcover", "cost"].includes(name)) syncLayerTabs();
        }
      });
    });

    $("layer-tabs").querySelectorAll(".chip").forEach((chip) => {
      chip.addEventListener("click", () => setBaseLayer(chip.dataset.base));
    });

    $("tool-zoom-in").addEventListener("click", () => map.zoomBy(1.4));
    $("tool-zoom-out").addEventListener("click", () => map.zoomBy(1 / 1.4));
    $("tool-fit").addEventListener("click", () => map.fit());
    $("tool-route").addEventListener("click", toggleRouteMode);
    $("route-cancel").addEventListener("click", () => toggleRouteMode(false));
    $("route-close").addEventListener("click", () => clearRoute());

    $("btn-compare").addEventListener("click", openCompare);
    $("btn-method").addEventListener("click", () => openModal("method-modal"));
    document.querySelectorAll("[data-close]").forEach((button) => button.addEventListener("click", closeModals));
    document.querySelectorAll(".modal").forEach((modal) => {
      modal.addEventListener("click", (event) => { if (event.target === modal) closeModals(); });
    });
  }

  function setBaseLayer(name) {
    ["resistance", "landcover", "cost"].forEach((layer) => {
      state.layers[layer] = layer === name;
      const input = document.querySelector(`#layer-switches input[data-layer="${layer}"]`);
      if (input) input.checked = state.layers[layer];
      map.setLayer(layer, state.layers[layer]);
    });
    syncLayerTabs();
    buildLegend(name);
  }

  function syncLayerTabs() {
    const active = ["resistance", "landcover", "cost"].find((name) => state.layers[name]) || null;
    $("layer-tabs").querySelectorAll(".chip").forEach((chip) => chip.classList.toggle("active", chip.dataset.base === active));
  }

  function buildLegend(kind) {
    const legend = $("legend");
    const title = legend.querySelector(".legend-title");
    const bar = $("legend-bar");
    if (kind === "landcover") {
      legend.style.display = "none";
      return;
    }
    legend.style.display = "";
    if (kind === "cost") {
      title.textContent = "Стоимость доступа";
      bar.style.background = "linear-gradient(90deg,#6eeebe,#4aaac2,#566cbe,#804a9e,#2e1e46)";
      legend.querySelector(".legend-ticks").innerHTML = "<span>0</span><span>25 %</span><span>50 %</span><span>100 %</span>";
    } else {
      title.textContent = "Сопротивление";
      bar.style.background = "linear-gradient(90deg,#0f3d33,#2b6e52,#7fa05e,#d8c173,#e08a4a,#c9503f,#8e2f3e)";
      legend.querySelector(".legend-ticks").innerHTML = "<span>1</span><span>10</span><span>100</span><span>1000</span>";
    }
  }

  /* ------------------------------------------------------------- analysis */
  function scheduleRun() {
    clearTimeout(runTimer);
    $("status-dot").classList.add("busy");
    setStatus("Параметры изменены — пересчёт…");
    runTimer = setTimeout(() => runAnalysis(state.scenario), 650);
  }

  async function runAnalysis(scenarioId) {
    clearTimeout(runTimer);
    if (state.busy) return;
    state.busy = true;
    state.requestSeq += 1;
    const seq = state.requestSeq;
    const params = readParams();
    state.params = params;
    const started = performance.now();

    setLoading(true, scenarioId === "base" ? "Расчёт поверхности сопротивления и коридоров…" : "Пересчёт сценария…");
    setStatus("Анализ сети…");
    try {
      const result = await API.analyze(scenarioId, params);
      if (seq !== state.requestSeq) return;
      state.result = result;
      applyResult(result);
      const seconds = ((performance.now() - started) / 1000).toFixed(1);
      $("status-elapsed").textContent = `расчёт ${seconds} с`;
      const cores = result.cores.features.length;
      setStatus(`Сеть: ${result.network.stats.link_count} ${plural(result.network.stats.link_count, "связь", "связи", "связей")} · ${fmt(result.network.stats.total_length_km, 0)} км · ${cores} ${plural(cores, "ядро", "ядра", "ядер")}`);
    } catch (error) {
      console.error(error);
      toast("Ошибка расчёта: " + error.message, true);
      setStatus("Ошибка расчёта");
    } finally {
      if (seq === state.requestSeq) {
        state.busy = false;
        setLoading(false);
        $("status-dot").classList.remove("busy");
      }
    }
  }

  function applyResult(result) {
    map.setImages({
      hillshade: result.layers.hillshade.url,
      landcover: result.layers.landcover.url,
      resistance: result.layers.resistance.url,
      cost: result.layers.cost.url,
    });
    map.setData(state.project, result);
    map.setLayer("resistance", state.layers.resistance);
    map.setLayer("landcover", state.layers.landcover);
    map.setLayer("cost", state.layers.cost);
    map.setLayer("hillshade", state.layers.hillshade);
    updatePanels(result);
    updateExportLinks(result.token);
    if (!map._sized) map.fit();
  }

  function updateExportLinks(token) {
    $("btn-report").href = API.reportUrl(token);
    $("btn-geojson").href = API.geojsonUrl(token);
    $("btn-report").setAttribute("download", `ecocorridor-report-${state.scenario}.html`);
    $("btn-geojson").setAttribute("download", `ecocorridor-corridors-${state.scenario}.geojson`);
  }

  /* --------------------------------------------------------------- panels */
  function updatePanels(result) {
    const frag = result.fragmentation || {};
    const stats = result.network.stats || {};
    const delta = result.delta || {};

    $("kpi-mesh").textContent = fmt(frag.mesh_ha, 0);
    $("kpi-pc").textContent = fmt(frag.pc, 3);
    $("kpi-iic").textContent = fmt(frag.iic, 3);
    $("kpi-links").textContent = fmt(stats.link_count, 0);

    setDelta("kpi-mesh-d", delta.mesh_ha_pct);
    setDelta("kpi-pc-d", delta.pc_pct);
    setDelta("kpi-iic-d", delta.iic_pct);
    setDelta("kpi-links-d", null);

    $("net-length").textContent = `${fmt(stats.total_length_km, 0)} км`;
    $("net-area").textContent = `${fmt(stats.corridor_area_ha, 0)} га`;
    $("net-cross").textContent = fmt(stats.road_crossings, 0);
    $("net-red").textContent = `${Math.round((stats.redundancy || 0) * 100)} %`;
    $("net-tag").textContent = `${stats.backbone_links} + ${stats.alternative_links}`;

    $("frag-patches").textContent = fmt(frag.patch_count, 0);
    $("frag-largest").textContent = `${fmt(frag.largest_patch_ha, 0)} га`;
    $("frag-edge").textContent = `${fmt(frag.edge_density_m_per_ha, 2)} м/га`;
    $("frag-division").textContent = fmt(frag.division, 3);
    if (frag.size_histogram) {
      $("patch-hist").innerHTML = Charts.histogram(frag.size_histogram.bins, frag.size_histogram.counts);
    }

    renderCorridorList(result);
    renderPinchList(result);
  }

  function setDelta(id, value) {
    const node = $(id);
    if (value === null || value === undefined || !Number.isFinite(value) || Math.abs(value) < 0.05) {
      node.textContent = value === null || value === undefined ? "" : "±0 %";
      node.className = "kpi-delta flat";
      return;
    }
    node.textContent = `${value > 0 ? "▲" : "▼"} ${Math.abs(value).toFixed(1)} %`;
    node.className = "kpi-delta " + (value > 0 ? "up" : "down");
  }

  function renderCorridorList(result) {
    const container = $("corridor-list");
    const links = result.network.links || [];
    if (!links.length) {
      container.innerHTML = `<p class="hint">В пределах заданной дистанции связи между ядрами не найдены.</p>`;
      return;
    }
    const names = new Map((result.cores.features || []).map((f) => [f.properties.core_id, shortName(f.properties.name)]));
    container.innerHTML = links
      .map((link) => {
        const label = `${names.get(link.from) || link.from} — ${names.get(link.to) || link.to}`;
        return `<div class="corridor" data-id="${link.id}" data-type="${link.type}">
          <span class="dot ${link.type}"></span>
          <span class="lbl" title="${label}">${label}</span>
          <span class="len">${fmt(link.length_km, 1)} км</span>
          <span class="cost">${fmt(link.cost / 1000, 0)} k</span>
        </div>`;
      })
      .join("");

    container.querySelectorAll(".corridor").forEach((node) => {
      const link = links.find((l) => String(l.id) === node.dataset.id);
      node.addEventListener("mouseenter", () => map.setHighlight({ type: "corridor", id: "l" + link.id, item: link }));
      node.addEventListener("mouseleave", () => map.setHighlight(null));
      node.addEventListener("click", () => focusLink(link, result));
    });
  }

  function focusLink(link, result) {
    const feature = result.corridors.features.find((f) => String(f.properties.id) === String(link.id));
    if (!feature) return;
    const coords = flattenCoords(feature.geometry);
    const xs = coords.map((c) => c[0]);
    const ys = coords.map((c) => c[1]);
    const padX = (Math.max(...xs) - Math.min(...xs)) * 0.35 + 800;
    const padY = (Math.max(...ys) - Math.min(...ys)) * 0.35 + 800;
    map.focusBounds([Math.min(...xs) - padX, Math.min(...ys) - padY, Math.max(...xs) + padX, Math.max(...ys) + padY]);
    map.setHighlight({ type: "corridor", id: "l" + link.id, item: link });
  }

  function renderPinchList(result) {
    const container = $("pinch-list");
    const pins = result.network.nodes ? result.pinchpoints.features : [];
    const features = result.pinchpoints.features || [];
    $("pinch-tag").textContent = String(features.length);
    if (!features.length) {
      container.innerHTML = `<p class="hint">Узкие места не обнаружены.</p>`;
      return;
    }
    container.innerHTML = features
      .map((feature, index) => {
        const p = feature.properties;
        return `<div class="pinch" data-index="${index}">
          <span>Узкое место ${index + 1}</span>
          <span class="meta">${fmt(p.width_m, 0)} м · поток ${fmt(p.flow, 1)}</span>
          <span class="flowbar" style="width:${Math.min(100, 30 + p.flow * 22)}%"></span>
        </div>`;
      })
      .join("");
    container.querySelectorAll(".pinch").forEach((node) => {
      const feature = features[Number(node.dataset.index)];
      const coords = feature.geometry.coordinates;
      node.addEventListener("mouseenter", () => map.setHighlight({ type: "pinch", id: "p" + node.dataset.index, item: feature }));
      node.addEventListener("mouseleave", () => map.setHighlight(null));
      node.addEventListener("click", () => {
        const pad = 3200;
        map.focusBounds([coords[0] - pad, coords[1] - pad, coords[0] + pad, coords[1] + pad]);
        map.setHighlight({ type: "pinch", id: "p" + node.dataset.index, item: feature });
      });
    });
  }

  function flattenCoords(geometry) {
    if (!geometry) return [];
    if (geometry.type === "LineString") return geometry.coordinates;
    if (geometry.type === "MultiLineString") return geometry.coordinates.flat();
    if (geometry.type === "Polygon") return geometry.coordinates[0];
    if (geometry.type === "MultiPolygon") return geometry.coordinates.flat(2);
    return [];
  }

  /* ------------------------------------------------------------------ map */
  function bindMap() {
    map.onView = (view) => {
      const target = 120;
      const raw = target / view.scale;
      const step = map._niceStep(raw);
      const px = step * view.scale;
      $("scalebar-line").style.width = `${Math.round(px)}px`;
      $("scalebar-label").textContent = step >= 1000 ? `${fmt(step / 1000, 0)} км` : `${fmt(step, 0)} м`;
    };

    map.onHover = (hit, screen, world) => {
      const tip = $("tooltip");
      if (hit) {
        const html = describeHit(hit);
        if (html) {
          tip.innerHTML = html;
          tip.style.left = `${screen[0]}px`;
          tip.style.top = `${screen[1]}px`;
          tip.classList.add("visible");
        }
      } else {
        tip.classList.remove("visible");
      }
      if (world) {
        $("readout-x").textContent = `${fmt(world[0] / 1000, 1)} / ${fmt(world[1] / 1000, 1)} км`;
        const lonlat = toLonLat(world[0], world[1]);
        $("readout-lonlat").textContent = `${lonlat[1].toFixed(4)}° N, ${lonlat[0].toFixed(4)}° E`;
        scheduleSample(world);
      }
    };

    map.onPick = (world) => {
      if (!state.routeMode) return;
      if (!state.routeStart) {
        state.routeStart = world;
        $("route-text").textContent = "Кликните конечную точку";
        map.setRoute({ coords: [world], points: [world] });
      } else {
        buildRoute(state.routeStart, world);
      }
    };

    map.resize();
  }

  function describeHit(hit) {
    if (hit.type === "core") {
      const p = hit.item;
      return `<b>${p.name}</b><br>${fmt(p.area_ha, 0)} га · R ${fmt(p.mean_resistance, 1)}<br>посредничество ${fmt(p.betweenness, 2)}`;
    }
    if (hit.type === "corridor") {
      const p = hit.item;
      const type = p.type === "backbone" ? "опорный" : "альтернативный";
      return `<b>${type} коридор</b><br>${fmt(p.length_km, 1)} км · стоимость ${fmt(p.cost / 1000, 0)} k<br>R ${fmt(p.mean_resistance, 1)} · общих участков ${fmt(p.shared_pct, 0)} %`;
    }
    if (hit.type === "pinch") {
      const p = hit.item;
      return `<b>Узкое место</b><br>ширина ${fmt(p.width_m, 0)} м · поток ${fmt(p.flow, 1)}`;
    }
    return "";
  }

  function toLonLat(x, y) {
    const corners = state.project?.extent_lonlat;
    if (!corners) return [0, 0];
    const [l, b, r, t] = state.project.bounds;
    const u = Math.min(1, Math.max(0, (x - l) / (r - l)));
    const v = Math.min(1, Math.max(0, (y - b) / (t - b)));
    const [sw, se, ne, nw] = corners;
    const lon = sw[0] * (1 - u) * (1 - v) + se[0] * u * (1 - v) + ne[0] * u * v + nw[0] * (1 - u) * v;
    const lat = sw[1] * (1 - u) * (1 - v) + se[1] * u * (1 - v) + ne[1] * u * v + nw[1] * (1 - u) * v;
    return [lon, lat];
  }

  function scheduleSample(world) {
    if (!state.result) return;
    clearTimeout(state.sampleTimer);
    state.sampleTimer = setTimeout(async () => {
      try {
        const sample = await API.sample(state.result.token, world[0], world[1]);
        $("readout-r").textContent = `R ${sample.resistance.toFixed(1)} · ${sample.class_name}`;
      } catch (_) {
        $("readout-r").textContent = "R —";
      }
    }, 130);
  }

  /* --------------------------------------------------------------- routes */
  function toggleRouteMode(force) {
    state.routeMode = force === undefined ? !state.routeMode : force;
    $("tool-route").classList.toggle("active", state.routeMode);
    $("map").classList.toggle("route-mode", state.routeMode);
    $("route-banner").classList.toggle("visible", state.routeMode);
    if (state.routeMode) {
      clearRoute();
      $("route-text").textContent = "Кликните начальную точку";
    } else {
      clearRoute();
    }
  }

  function clearRoute() {
    state.routeStart = null;
    map.clearRoute();
    map.setRoute(null);
    $("route-summary").classList.remove("visible");
    if (state.routeMode) $("route-text").textContent = "Кликните начальную точку";
  }

  async function buildRoute(start, end) {
    if (!state.result) return;
    setStatus("Построение маршрута…");
    try {
      const route = await API.route(state.result.token, start, end);
      route.points = [start, end];
      map.setRoute(route);
      $("route-text").textContent = "Готово. Кликните, чтобы начать новый маршрут";
      state.routeStart = null;
      $("rs-length").textContent = `${fmt(route.length_km, 1)} км`;
      $("rs-cost").textContent = fmt(route.cost / 1000, 0) + " k";
      $("rs-mean").textContent = fmt(route.mean_resistance, 1);
      $("rs-straight").textContent = `${fmt(route.straight_km, 1)} км`;
      const detour = route.length_km > 0 ? ((route.length_km / route.straight_km - 1) * 100).toFixed(0) : "0";
      $("rs-note").textContent = Number(detour) > 3
        ? `Путь на ${detour} % длиннее прямой и обходит участки с высоким сопротивлением.`
        : "Прямой путь почти не встречает барьеров.";
      $("route-summary").classList.add("visible");
      setStatus(`Маршрут: ${fmt(route.length_km, 1)} км, стоимость ${fmt(route.cost / 1000, 0)} k`);
    } catch (error) {
      toast("Маршрут не найден: " + error.message, true);
    }
  }

  /* -------------------------------------------------------------- compare */
  async function openCompare() {
    openModal("compare-modal");
    const body = $("compare-body");
    body.innerHTML = `<div class="compare-loading"><div class="spinner"></div>Пересчёт сценариев…</div>`;
    try {
      const scenarios = state.project.scenarios.map((s) => s.id);
      const rows = await API.compare(scenarios, readParams());
      state.compareRows = rows.rows;
      body.innerHTML = Charts.compareBody(rows.rows, state.scenario);
      body.querySelectorAll("tbody tr").forEach((row, index) => {
        row.style.cursor = "pointer";
        row.addEventListener("click", () => {
          const scenario = rows.rows[index];
          closeModals();
          selectScenario(scenario.id);
        });
      });
    } catch (error) {
      body.innerHTML = `<p class="compare-note">Не удалось сравнить сценарии: ${error.message}</p>`;
    }
  }

  /* --------------------------------------------------------------- modals */
  function openModal(id) {
    document.querySelectorAll(".modal").forEach((modal) => modal.classList.toggle("open", modal.id === id));
  }

  function closeModals() {
    document.querySelectorAll(".modal").forEach((modal) => modal.classList.remove("open"));
  }

  function bindShortcuts() {
    window.addEventListener("keydown", (event) => {
      if (event.target.matches("input, select, textarea")) return;
      if (event.key === "Escape") {
        closeModals();
        if (state.routeMode) toggleRouteMode(false);
      } else if (event.key === "f" || event.key === "F" || event.key === "а") {
        map.fit();
      } else if (event.key === "r" || event.key === "R" || event.key === "к") {
        toggleRouteMode();
      }
    });
  }

  return { init };
})();

document.addEventListener("DOMContentLoaded", App.init);
