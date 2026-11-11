/* EcoCorridor — "Загрузить территорию" dialog */

const Territory = (() => {
  const $ = (id) => document.getElementById(id);
  const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const st = { places: [], picked: null, planTimer: null, planSeq: 0, loading: false };

  function request() {
    const body = {
      core_source: $("t-cores").value,
      min_core_ha: Number($("t-mincore").value) || 100,
    };
    const size = Number($("t-size").value);
    if (size) body.size_km = size;
    const res = $("t-res").value;
    if (res) body.resolution_m = Number(res);
    if (st.picked) {
      body.query = st.picked.query;
      body.place_index = st.picked.index;
      body.name = st.picked.name;
    } else {
      const lon = parseFloat($("t-lon").value);
      const lat = parseFloat($("t-lat").value);
      if (!Number.isFinite(lon) || !Number.isFinite(lat)) return null;
      body.center = [lon, lat];
      body.name = `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"} ${Math.abs(lon).toFixed(2)}°${lon >= 0 ? "E" : "W"}`;
    }
    return body;
  }

  /* ------------------------------------------------------------- search */
  async function search(event) {
    event.preventDefault();
    const q = $("t-query").value.trim();
    if (q.length < 2) return;
    const box = $("t-places");
    box.innerHTML = `<p class="hint">Ищу «${esc(q)}»…</p>`;
    $("t-search-btn").disabled = true;
    try {
      const { places } = await API.searchPlace(q);
      st.places = places;
      st.query = q;
      if (!places.length) {
        box.innerHTML = `<p class="hint">Ничего не найдено. Уточните название или задайте центр координатами.</p>`;
        return;
      }
      box.innerHTML = places
        .map((p, i) => `<button type="button" class="t-place" data-i="${i}"><b>${esc(p.name)}</b>
          <span>${esc(p.display_name)} · ${p.width_km} × ${p.height_km} км</span></button>`)
        .join("");
      box.querySelectorAll(".t-place").forEach((node) => node.addEventListener("click", () => pick(Number(node.dataset.i))));
      if (places.length === 1) pick(0);
    } catch (error) {
      box.innerHTML = `<p class="hint t-err">Поиск не удался: ${esc(error.message)}</p>`;
    } finally {
      $("t-search-btn").disabled = false;
    }
  }

  function pick(i) {
    const p = st.places[i];
    st.picked = { query: st.query, index: i, name: p.name, place: p };
    document.querySelectorAll("#t-places .t-place").forEach((n) => n.classList.toggle("active", Number(n.dataset.i) === i));
    $("t-options").hidden = false;
    $("t-selected").innerHTML = `<b>${esc(p.name)}</b><small>${esc(p.display_name)}</small>`;
    // Start from the place's own extent, within what the server accepts.
    const side = Math.max(p.width_km, p.height_km);
    $("t-size").value = String(Math.min(122, Math.max(20, Math.round(side * 1.25 / 5) * 5 || 50)));
    schedulePlan();
  }

  /* --------------------------------------------------------------- plan */
  function schedulePlan() {
    clearTimeout(st.planTimer);
    st.planTimer = setTimeout(plan, 350);
  }

  async function plan() {
    const body = request();
    const box = $("t-plan");
    if (!body) {
      box.textContent = "";
      return;
    }
    const seq = ++st.planSeq;
    try {
      const result = await API.planTerritory(body);
      if (seq !== st.planSeq) return;
      const e = result.extent;
      box.innerHTML =
        `Кадр <b>${e.width_km} × ${e.height_km} км</b> (${Math.round(e.area_km2).toLocaleString("ru-RU")} км²), ` +
        `ячейка <b>${e.resolution_m} м</b>, ${e.cells.toLocaleString("ru-RU")} ячеек, ${e.crs}.` +
        e.warnings.map((w) => `<span class="warn">${esc(w)}</span>`).join("");
      st.projectId = result.project_id;
      refreshCached();
    } catch (error) {
      if (seq === st.planSeq) box.innerHTML = `<span class="err">${esc(error.message)}</span>`;
    }
  }

  /* --------------------------------------------------------------- load */
  function setProgress(visible, fraction = 0, text = "") {
    $("t-progress").hidden = !visible;
    $("t-bar-fill").style.width = `${Math.round(fraction * 100)}%`;
    $("t-progress-text").textContent = text;
  }

  async function load() {
    const body = request();
    if (!body || st.loading) return;
    body.force = $("t-force").checked;
    st.loading = true;
    $("t-load").disabled = true;
    setProgress(true, 0.02, "Запуск…");
    try {
      const { job_id } = await API.loadTerritory(body);
      for (;;) {
        await new Promise((r) => setTimeout(r, 700));
        const job = await API.job(job_id);
        if (job.status === "error") throw new Error(job.error || "ошибка загрузки");
        setProgress(true, job.fraction || 0, job.message || "");
        if (job.status === "done") {
          App.closeModals();
          App.toast("Территория загружена");
          await App.openProject(job.project_id);
          break;
        }
      }
    } catch (error) {
      setProgress(true, 0, "");
      $("t-progress-text").innerHTML = `<span class="err" style="color:var(--red)">${esc(error.message)}</span>`;
    } finally {
      st.loading = false;
      $("t-load").disabled = false;
      refreshCached();
    }
  }

  /* ------------------------------------------------------------- cached */
  async function refreshCached() {
    const box = $("t-cached");
    try {
      const { projects } = await API.projects();
      const current = App.currentId();
      box.innerHTML = projects
        .map((p) => {
          const meta = p.synthetic ? "демонстрационный, синтетический" : [p.area_km2 ? `${Math.round(p.area_km2).toLocaleString("ru-RU")} км²` : "", p.resolution_m ? `${p.resolution_m} м` : ""].filter(Boolean).join(" · ");
          return `<button type="button" class="t-place ${p.id === current ? "current" : ""}" data-id="${esc(p.id)}">
            <b>${esc(p.name)}</b><span>${esc(meta)}${p.id === current ? " · открыт" : ""}</span></button>`;
        })
        .join("");
      box.querySelectorAll(".t-place").forEach((node) =>
        node.addEventListener("click", async () => {
          App.closeModals();
          await App.openProject(node.dataset.id);
        })
      );
    } catch (_) {
      box.innerHTML = "";
    }
  }

  function bind() {
    $("t-search-form").addEventListener("submit", search);
    ["t-size", "t-res", "t-cores", "t-mincore", "t-lon", "t-lat"].forEach((id) => {
      $(id).addEventListener("input", () => {
        if (id === "t-lon" || id === "t-lat") {
          st.picked = null;
          $("t-options").hidden = false;
          $("t-selected").innerHTML = "<b>Заданный центр</b>";
        }
        schedulePlan();
      });
    });
    $("t-load").addEventListener("click", load);
  }

  document.addEventListener("DOMContentLoaded", bind);
  return { opened: refreshCached };
})();
