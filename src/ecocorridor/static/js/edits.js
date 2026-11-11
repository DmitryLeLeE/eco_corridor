/* EcoCorridor — scenario edits drawn on the map (road, built-up area, new forest, wildlife crossing) */

const Edits = (() => {
  const $ = (id) => document.getElementById(id);
  const esc = (t) => String(t).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const TYPES = {
    road: { name: "Новая дорога", line: true, color: "#b3261e", fill: "rgba(179,38,30,0.35)", need: 2 },
    builtup: { name: "Застройка", line: false, color: "#8a4b08", fill: "rgba(138,75,8,0.35)", need: 3 },
    habitat: { name: "Новый лес", line: false, color: "#1b6b3a", fill: "rgba(27,107,58,0.32)", need: 3 },
    crossing: { name: "Переход для животных", line: true, color: "#0b6e8a", fill: "rgba(11,110,138,0.38)", need: 2 },
  };

  const st = { list: [], draft: null, getMap: () => null, onChange: () => {} };

  const kindOf = () => $("edit-type").value;
  const widthOf = () => Math.min(2000, Math.max(10, Number($("edit-width").value) || 60));

  function sync() {
    const map = st.getMap();
    if (map) {
      map.edits = st.list.map((e) => ({ ...e, style: TYPES[e.type] }));
      map.draft = st.draft ? { ...st.draft, style: TYPES[st.draft.type] } : null;
      map.drawing = !!st.draft;
      map.rasterDirty = true;
    }
    $("edit-list").innerHTML = st.list
      .map(
        (e, i) => `<li><i style="background:${TYPES[e.type].color}"></i><span>${esc(TYPES[e.type].name)}${TYPES[e.type].line ? `, ${e.width_m} м` : ""}</span>
          <button type="button" class="link-btn" data-del="${i}" title="Удалить">×</button></li>`
      )
      .join("");
    $("edit-list").querySelectorAll("[data-del]").forEach((b) =>
      b.addEventListener("click", () => {
        st.list.splice(Number(b.dataset.del), 1);
        sync();
        st.onChange();
      })
    );
    $("edit-count").textContent = st.list.length ? String(st.list.length) : "";
    $("edit-clear").hidden = !st.list.length;
    $("edit-width-field").hidden = !TYPES[kindOf()].line;
    const banner = $("edit-banner");
    banner.classList.toggle("visible", !!st.draft);
    if (st.draft) {
      const need = TYPES[st.draft.type].need;
      $("edit-text").textContent = `${TYPES[st.draft.type].name}: точек ${st.draft.coords.length}. ` +
        (st.draft.coords.length >= need ? "Enter или «готово» — завершить." : `Нужно минимум ${need}.`);
    }
    $("edit-draw").classList.toggle("active", !!st.draft);
  }

  function start() {
    if (st.draft) return finish(false);
    const type = kindOf();
    st.draft = { type, coords: [], width_m: widthOf() };
    sync();
  }

  function pick(world) {
    if (!st.draft) return false;
    const map = st.getMap();
    const last = st.draft.coords[st.draft.coords.length - 1];
    if (last && map) {
      const a = map.toScreen(last);
      const b = map.toScreen(world);
      if (Math.hypot(a[0] - b[0], a[1] - b[1]) < 5) return true; // second click of a double click
    }
    st.draft.coords.push([Math.round(world[0] * 10) / 10, Math.round(world[1] * 10) / 10]);
    sync();
    return true;
  }

  function finish(commit = true) {
    const draft = st.draft;
    st.draft = null;
    if (commit && draft && draft.coords.length >= TYPES[draft.type].need) {
      st.list.push(draft);
      sync();
      st.onChange();
    } else {
      if (commit && draft && draft.coords.length) App.toast(`Мало точек: нужно минимум ${TYPES[draft.type].need}`, true);
      sync();
    }
  }

  function undoPoint() {
    if (st.draft && st.draft.coords.length) {
      st.draft.coords.pop();
      sync();
    }
  }

  function init({ getMap, onChange }) {
    st.getMap = getMap;
    st.onChange = onChange;
    $("edit-draw").addEventListener("click", start);
    $("edit-done").addEventListener("click", () => finish(true));
    $("edit-cancel").addEventListener("click", () => finish(false));
    $("edit-clear").addEventListener("click", () => {
      st.list = [];
      sync();
      st.onChange();
    });
    $("edit-type").addEventListener("change", sync);
    window.addEventListener("keydown", (event) => {
      if (!st.draft || event.target.matches("input, select, textarea")) return;
      if (event.key === "Enter") finish(true);
      else if (event.key === "Escape") finish(false);
      else if (event.key === "Backspace") undoPoint();
    });
    sync();
  }

  return {
    init,
    pick,
    sync,
    active: () => !!st.draft,
    list: () => st.list.map((e) => ({ type: e.type, coords: e.coords, width_m: e.width_m })),
    reset: () => {
      st.list = [];
      st.draft = null;
      sync();
    },
  };
})();
