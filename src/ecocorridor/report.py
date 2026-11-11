"""Self-contained HTML report (map + metrics + methodology)."""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime

import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

from . import render
from .analysis import AnalysisResult
from .dataset import Project


def _fmt(value, digits: int = 0) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}".replace(",", " ")
    if isinstance(value, (float, np.floating)):
        text = f"{float(value):,.{digits}f}"
        return text.replace(",", " ").replace(".", ",")
    return html.escape(str(value))


def _map_svg(project: Project, result: AnalysisResult, width: int = 1180) -> str:
    left, bottom, right, top = project.bounds
    height = int(width * (top - bottom) / (right - left))

    def pt(x: float, y: float) -> tuple[float, float]:
        return ((x - left) / (right - left) * width, (top - y) / (top - bottom) * height)

    rgba = render.resistance_rgba(project, result.resistance)
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").resize((width, height), Image.LANCZOS).save(buffer, format="PNG", optimize=True)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    parts: list[str] = [f'<image href="data:image/png;base64,{image_b64}" x="0" y="0" width="{width}" height="{height}"/>']

    parts.append('<g fill="#58e0a8" fill-opacity="0.10" stroke="#58e0a8" stroke-opacity="0.45" stroke-width="1">')
    for geom in result.network.polygons.geometry:
        for poly in _polygons(geom):
            parts.append(_polygon_path(poly, pt))
    parts.append("</g>")

    parts.append('<g fill="#6ff0b4" fill-opacity="0.16" stroke="#8dffc6" stroke-width="2">')
    for geom in result.cores.geometry:
        for poly in _polygons(geom):
            parts.append(_polygon_path(poly, pt))
    parts.append("</g>")

    parts.append('<g fill="none" stroke-linecap="round" stroke-linejoin="round">')
    for row in result.network.lines.itertuples():
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(*c) for c in row.geometry.coords))
        color = "#0e7a52" if row.type == "backbone" else "#8a6a2f"
        parts.append(f'<polyline points="{path}" stroke="#eafff5" stroke-opacity="0.65" stroke-width="4.5"/>')
        parts.append(f'<polyline points="{path}" stroke="{color}" stroke-width="2.4"/>')
    parts.append("</g>")

    parts.append('<g>')
    for row in result.network.pinchpoints.itertuples():
        x, y = pt(row.geometry.x, row.geometry.y)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#ffd166" stroke="#3a2a00" stroke-width="1.2"/>')
    parts.append("</g>")

    label_style = (
        'font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="600" '
        'fill="#ffffff" stroke="#0b1512" stroke-width="3.2" paint-order="stroke" stroke-linejoin="round"'
    )
    parts.append(f"<g {label_style}>")
    for row in result.cores.itertuples():
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        anchor = _label_anchor(project, cx, cy)
        x, y = pt(cx, cy)
        parts.append(f'<text x="{x:.0f}" y="{y:.0f}" text-anchor="middle">{html.escape(str(row.name))}</text>')
    parts.append("</g>")

    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:auto;border-radius:10px;">' + "".join(parts) + "</svg>"
    )


def _label_anchor(project: Project, x: float, y: float) -> str:
    return "middle"


def _polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if hasattr(geom, "geoms"):
        out = []
        for part in geom.geoms:
            out.extend(_polygons(part))
        return out
    return []


def _polygon_path(poly: Polygon, pt) -> str:
    coords = list(poly.exterior.coords)
    if len(coords) < 3:
        return ""
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(*c) for c in coords))
    return f'<polygon points="{points}"/>'


def _legend_html() -> str:
    stops = render.RESISTANCE_STOPS
    css = ", ".join(f'{("#%02x%02x%02x" % c)} {p * 100:.0f}%' for p, c in stops)
    ticks = [1, 5, 20, 100, 500, 2000]
    labels = "".join(f"<span>{t}</span>" for t in ticks)
    return (
        f'<div class="legend-bar" style="background:linear-gradient(90deg,{css})"></div>'
        f'<div class="legend-ticks">{labels}</div>'
    )


def build_report(project: Project, result: AnalysisResult) -> str:
    frag = result.fragmentation
    stats = result.network.stats
    generated = datetime.now().strftime("%d.%m.%Y %H:%M")
    cores = result.cores
    map_svg = _map_svg(project, result)

    corridor_rows = "".join(
        f"<tr><td>{html.escape(str(row.id))}</td><td>{'опорный' if row.type == 'backbone' else 'альтернативный'}</td>"
        f"<td>{_fmt(row.length_km, 1)}</td><td>{_fmt(row.cost, 0)}</td>"
        f"<td>{_fmt(row.mean_resistance, 1)}</td><td>{_fmt(row.shared_pct, 0)}</td></tr>"
        for row in result.network.lines.itertuples()
    )

    pinch_rows = "".join(
        f"<tr><td>{_fmt(row.flow, 1)}</td><td>{_fmt(row.width_m, 0)}</td>"
        f"<td>{_fmt(row.geometry.y, 0)} / {_fmt(row.geometry.x, 0)}</td></tr>"
        for row in result.network.pinchpoints.itertuples()
    )

    core_rows = "".join(
        f"<tr><td>{html.escape(str(row.name))}</td><td>{_fmt(row.area_ha, 0)}</td>"
        f"<td>{_fmt(row.mean_resistance, 1)}</td><td>{_fmt(row.betweenness, 2)}</td></tr>"
        for row in cores.itertuples()
    )

    summary = result.scenario_summary
    scenario_note = ""
    if result.scenario.id == "urban":
        scenario_note = f"Под застройку перешло {_fmt(summary.get('converted_ha'), 0)} га."
    elif result.scenario.id == "road":
        scenario_note = f"Новая трасса занимает {_fmt(summary.get('road_ha'), 0)} га, сопротивление на её полосе — не ниже 400."
    elif result.scenario.id == "restore":
        scenario_note = f"Восстановлено {_fmt(summary.get('restored_ha'), 0)} га пойменных местообитаний."

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>EcoCorridor — отчёт «{html.escape(project.name)}»</title>
<style>
  :root {{ --ink:#15221d; --muted:#5d6f66; --line:#d9e2dc; --accent:#0e7a52; --bg:#fbfaf6; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font-family:"Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height:1.5; }}
  .page {{ max-width:1240px; margin:0 auto; padding:44px 40px 64px; }}
  header {{ display:flex; justify-content:space-between; align-items:flex-end; gap:24px;
            border-bottom:2px solid var(--ink); padding-bottom:18px; }}
  .brand {{ font-size:13px; letter-spacing:.18em; text-transform:uppercase; color:var(--accent); font-weight:700; }}
  h1 {{ font-size:34px; line-height:1.15; margin:8px 0 6px; letter-spacing:-.01em; }}
  .sub {{ color:var(--muted); font-size:15px; max-width:720px; }}
  .stamp {{ text-align:right; font-size:12px; color:var(--muted); white-space:nowrap; }}
  .kpis {{ display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin:26px 0 8px; }}
  .kpi {{ border:1px solid var(--line); background:#fff; border-radius:10px; padding:14px 16px; }}
  .kpi b {{ display:block; font-size:26px; font-weight:700; letter-spacing:-.02em; }}
  .kpi span {{ font-size:11.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }}
  figure {{ margin:26px 0 10px; border:1px solid var(--line); background:#0d1412; border-radius:12px; overflow:hidden; }}
  .map-note {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-start; padding:12px 16px; background:#fff; }}
  .legend {{ min-width:280px; }}
  .legend-bar {{ height:10px; border-radius:6px; }}
  .legend-ticks {{ display:flex; justify-content:space-between; font-size:10.5px; color:var(--muted); margin-top:3px; }}
  .map-legend {{ font-size:12px; color:var(--muted); display:flex; gap:18px; flex-wrap:wrap; }}
  .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }}
  h2 {{ font-size:19px; margin:34px 0 12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13.5px; background:#fff; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }}
  th {{ font-size:11.5px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); font-weight:600; }}
  td.num {{ font-variant-numeric:tabular-nums; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:26px; }}
  .method {{ background:#f2f4ef; border-left:3px solid var(--accent); padding:16px 20px; border-radius:0 8px 8px 0;
             font-size:13.5px; color:#33443c; }}
  .method code {{ background:#e6ebe3; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
  footer {{ margin-top:40px; border-top:1px solid var(--line); padding-top:14px; font-size:11.5px; color:var(--muted);
            display:flex; justify-content:space-between; gap:20px; }}
  @media print {{ body {{ background:#fff; }} .page {{ padding:18px; }} figure {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<div class="page">
  <header>
    <div>
      <div class="brand">EcoCorridor · анализ связности ландшафтов</div>
      <h1>{html.escape(project.name)}</h1>
      <p class="sub">Сценарий: <b>{html.escape(result.scenario.name)}</b>. {html.escape(result.scenario.description)} {scenario_note}</p>
    </div>
    <div class="stamp">Отчёт сформирован<br><b>{generated}</b><br>{html.escape(project.crs)} · {_fmt(project.res)} м/пиксель</div>
  </header>

  <div class="kpis">
    <div class="kpi"><span>Эффективный размер, MESH</span><b>{_fmt(frag.get('mesh_ha'), 0)}<small> га</small></b></div>
    <div class="kpi"><span>Вероятностная связность, PC</span><b>{_fmt(frag.get('pc'), 3)}</b></div>
    <div class="kpi"><span>Индекс IIC</span><b>{_fmt(frag.get('iic'), 3)}</b></div>
    <div class="kpi"><span>Длина сети коридоров</span><b>{_fmt(stats.get('total_length_km'), 0)}<small> км</small></b></div>
    <div class="kpi"><span>Узких мест</span><b>{_fmt(stats.get('pinchpoint_count'))}</b></div>
  </div>

  <figure>
    {map_svg}
    <div class="map-note">
      <div class="legend">
        <div style="font-size:11.5px;color:var(--muted);margin-bottom:4px;">Сопротивление перемещению (log-шкала)</div>
        {_legend_html()}
      </div>
      <div class="map-legend">
        <span><i class="dot" style="background:#8dffc6"></i>ядра ООПТ</span>
        <span><i class="dot" style="background:#0e7a52"></i>опорные коридоры</span>
        <span><i class="dot" style="background:#8a6a2f"></i>альтернативные</span>
        <span><i class="dot" style="background:#ffd166"></i>узкие места</span>
      </div>
    </div>
  </figure>

  <div class="grid2">
    <div>
      <h2>Структура ландшафта</h2>
      <table>
        <tr><th>Показатель</th><th>Значение</th></tr>
        <tr><td>Патчей местообитаний (≥ {_fmt(frag.get('min_patch_ha'), 1)} га)</td><td class="num">{_fmt(frag.get('patch_count'))}</td></tr>
        <tr><td>Площадь местообитаний</td><td class="num">{_fmt(frag.get('total_area_ha'))} га</td></tr>
        <tr><td>Крупнейший патч</td><td class="num">{_fmt(frag.get('largest_patch_ha'))} га</td></tr>
        <tr><td>Доля крупнейшего патча (LPI)</td><td class="num">{_fmt(frag.get('lpi'), 1)} %</td></tr>
        <tr><td>Средний размер патча</td><td class="num">{_fmt(frag.get('mean_patch_ha'))} га</td></tr>
        <tr><td>Эффективный размер сети (MESH)</td><td class="num">{_fmt(frag.get('mesh_ha'))} га</td></tr>
        <tr><td>Индекс дробности (division)</td><td class="num">{_fmt(frag.get('division'), 3)}</td></tr>
        <tr><td>Плотность границ</td><td class="num">{_fmt(frag.get('edge_density_m_per_ha'), 2)} м/га</td></tr>
        <tr><td>Доля крупнейшей связной компоненты</td><td class="num">{_fmt(frag.get('largest_component_share'), 3)}</td></tr>
      </table>
    </div>
    <div>
      <h2>Ядра сети</h2>
      <table>
        <tr><th>Территория</th><th>Площадь, га</th><th>Сопротивление</th><th>Посредничество</th></tr>
        {core_rows}
      </table>
      <h2>Узкие места</h2>
      <table>
        <tr><th>Поток</th><th>Ширина, м</th><th>Y / X, м</th></tr>
        {pinch_rows}
      </table>
    </div>
  </div>

  <h2>Коридорная сеть</h2>
  <table>
    <tr><th>Связь</th><th>Тип</th><th>Длина, км</th><th>Стоимость</th><th>Среднее сопротивление</th><th>Общие участки, %</th></tr>
    {corridor_rows}
  </table>

  <h2>Методика</h2>
  <div class="method">
    <p><b>Поверхность сопротивления.</b> Базовая стоимость класса земного покрова умножается на поправки за уклон,
    удалённость от дорог и застройки:
    <code>R = R<sub>lc</sub> · (1 + w<sub>s</sub>·s) · (1 + w<sub>r</sub>·e<sup>−d<sub>r</sub>/D<sub>r</sub></sup>) · (1 + w<sub>u</sub>·e<sup>−d<sub>u</sub>/D<sub>u</sub></sup>)</code>.
    Стоимость перехода между соседними ячейками — <code>d · (R<sub>a</sub> + R<sub>b</sub>)/2</code>.</p>
    <p><b>Наименьшая стоимость.</b> Взвешенный граф из 8-соседних ячеек, точный алгоритм Дейкстры. Сеть ядер строится
    как минимальное остовное дерево по стоимости связей; дополнительно добавляются альтернативные коридоры, если они
    не дороже остовного пути более чем в {_fmt(result.params.corridors.detour_ratio, 2)} раза.</p>
    <p><b>Связность.</b> IIC = ΣΣ a<sub>i</sub>a<sub>j</sub>/(1+n<sub>ij</sub>)/A², PC = ΣΣ a<sub>i</sub>a<sub>j</sub>p<sub>ij</sub>/A²
    с экспоненциальным ядром расселения (d₀ = {_fmt(frag.get('dispersal_m'), 0)} м). Узкие места — участки с высоким потоком
    связей и малой локальной шириной коридора.</p>
    <p><b>Ограничения.</b> Демонстрационный ландшафт синтетический; значения сопротивления относительны, а не
    откалиброваны по данным о перемещениях животных. Результат следует интерпретировать как сценарий
    относительного сравнения, а не как абсолютную оценку проницаемости.</p>
  </div>

  <footer>
    <span>EcoCorridor v{html.escape(str(__import__('ecocorridor').__version__))} · Rasterio, GeoPandas, SciPy, NetworkX</span>
    <span>Сценарий «{html.escape(result.scenario.name)}» · расчёт {_fmt(result.elapsed, 1)} с</span>
  </footer>
</div>
</body>
</html>"""
