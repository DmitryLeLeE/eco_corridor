/* EcoCorridor — formatting and chart helpers */

const Charts = (() => {
  const nf = (value, digits = 0) => {
    if (value === null || value === undefined || Number.isNaN(value)) return "—";
    return Number(value).toLocaleString("ru-RU", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  };

  const pct = (value) => {
    if (value === null || value === undefined) return "";
    const sign = value > 0 ? "+" : "";
    return `${sign}${nf(value, 1)} %`;
  };

  const deltaClass = (value) => (value === null || value === undefined || Math.abs(value) < 0.05 ? "flat" : value > 0 ? "up" : "down");

  function deltaChip(value, suffix = "%") {
    if (value === null || value === undefined) return "";
    const cls = deltaClass(value);
    const text = `${value > 0 ? "+" : ""}${nf(value, 1)}${suffix === "%" ? " %" : suffix}`;
    return `<span class="delta ${cls}">${text}</span>`;
  }

  function barRow(label, value, max, color, display, min = 0) {
    const width = Math.max(2, Math.min(100, ((value - min) / (max - min || 1)) * 100));
    return `
      <div class="bar-row">
        <span class="nm">${label}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${width}%;background:${color}"></div></div>
        <span class="val">${display ?? nf(value, 0)}</span>
      </div>`;
  }

  /** Bars centred on the base value: base = 50 %, window = ±window % . */
  function windowBars(rows, valueOf, format, windowPct = 5) {
    const base = valueOf(rows[0]) || 1;
    const max = base * (1 + windowPct / 100);
    const min = base * (1 - windowPct / 100);
    return rows
      .map((row) => {
        const value = valueOf(row);
        const width = Math.max(2, Math.min(100, ((value - min) / (max - min)) * 100));
        return `<div class="bar-row">
          <span class="nm">${row.name}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%;background:${row.accent || "#5be3a6"}"></div></div>
          <span class="val">${format(value)}</span>
        </div>`;
      })
      .join("");
  }

  function histogram(bins, counts) {
    const max = Math.max(...counts, 1);
    return counts
      .map((count, i) => `<div class="bar" style="height:${Math.max(3, (count / max) * 100)}%" title="${bins[i]}–${bins[i + 1]} га: ${count}"></div>`)
      .join("");
  }

  function compareBody(rows, activeId) {
    if (!rows?.length) return `<div class="compare-loading">Нет данных</div>`;
    const metrics = [
      { label: "MESH, га", window: 5, value: (row) => row.fragmentation?.mesh_ha ?? 0, format: (v) => nf(v, 0) },
      { label: "PC", window: 5, value: (row) => row.fragmentation?.pc ?? 0, format: (v) => nf(v, 3) },
      { label: "IIC", window: 5, value: (row) => row.fragmentation?.iic ?? 0, format: (v) => nf(v, 3) },
      { label: "Стоимость сети", window: 12, value: (row) => row.network?.network_cost ?? 0, format: (v) => nf(v / 1000, 0) + " k" },
      { label: "Длина сети, км", window: 12, value: (row) => row.network?.total_length_km ?? 0, format: (v) => nf(v, 0) },
    ];

    const blocks = metrics
      .map(
        (metric) => `<div class="compare-block">
          <h3>${metric.label}<span class="block-note">±${metric.window} % от базового</span></h3>
          <div class="bars">${windowBars(rows, metric.value, metric.format, metric.window)}</div>
        </div>`
      )
      .join("");

    const tableRows = rows.map((row) => {
      const f = row.fragmentation || {};
      const n = row.network || {};
      const isActive = row.id === activeId;
      return `<tr class="${isActive ? "active" : ""}">
        <td><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${row.accent};margin-right:7px"></span>${row.name}</td>
        <td>${nf(f.mesh_ha, 0)}</td>
        <td>${nf(f.pc, 3)}</td>
        <td>${nf(f.iic, 3)}</td>
        <td>${nf(f.patch_count, 0)}</td>
        <td>${nf(n.link_count, 0)}</td>
        <td>${nf(n.total_length_km, 0)}</td>
        <td>${nf((n.network_cost ?? 0) / 1000, 0)} k</td>
        <td>${nf(n.mean_resistance, 1)}</td>
        <td class="delta ${deltaClass(row.delta?.pc_pct)}">${row.id === "base" ? "—" : pct(row.delta?.pc_pct)}</td>
      </tr>`;
    }).join("");

    return `
      ${blocks}
      <div class="compare-block">
        <h3>Полная таблица</h3>
        <table class="compare-table">
          <thead><tr>
            <th>Сценарий</th><th>MESH, га</th><th>PC</th><th>IIC</th><th>Патчей</th>
            <th>Связей</th><th>Длина, км</th><th>Стоимость</th><th>R коридоров</th><th>Δ PC</th>
          </tr></thead>
          <tbody>${tableRows}</tbody>
        </table>
      </div>
      <p class="compare-note">Шкалы столбцов ограничены окном ±5 % (для стоимости и длины — ±12 %) от базового
        сценария, чтобы различия были читаемы. Урбанизация снижает MESH и PC: местообитания теряются, ядра
        становятся изолированнее. Новая трасса повышает стоимость сети, а восстановление пойм её снижает при той же
        топологии. Все сценарии пересчитаны при одинаковых параметрах модели.</p>`;
  }

  return { nf, pct, deltaClass, deltaChip, histogram, compareBody, barRow };
})();
