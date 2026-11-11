#!/usr/bin/env python
"""Build the GitHub Pages site into ``site/``.

* documentation: ``docs/*.md`` + ``CHANGELOG.md`` -> HTML (python-markdown)
* static demo: the bundled demo project is analysed for the four scenarios with the
  default parameters; the browser client in ``src/ecocorridor/static`` is copied and
  its API module is replaced by a shim that reads those pre-computed results
* provenance: ``site/provenance.json`` records versions, commit and build time

Usage::

    python scripts/build_pages.py [--out site]
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import markdown  # noqa: E402

from ecocorridor import __version__  # noqa: E402

SCENARIOS = ("base", "urban", "road", "restore")
SITE_URL = "https://dmitrylelee.github.io/eco_corridor/"

# page file -> (source markdown, nav title)
PAGES = {
    "index": ("docs/index.md", "Главная"),
    "guide": ("docs/guide.md", "Справка"),
    "methodology": ("docs/methodology.md", "Методика"),
    "validation": ("docs/validation.md", "Валидация"),
    "data": ("docs/data.md", "Данные"),
    "deploy": ("docs/deploy.md", "Размещение"),
    "reproducibility": ("docs/reproducibility.md", "Воспроизводимость"),
    "science": ("docs/science.md", "Научные принципы"),
    "review": ("docs/METHODOLOGY_REVIEW.md", "Аудит методики"),
    "references": ("docs/references.md", "Источники"),
    "changelog": ("CHANGELOG.md", "История изменений"),
}
LINKS = {Path(src).name: f"{name}.html" for name, (src, _t) in PAGES.items()}


# --------------------------------------------------------------------------- helpers
def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def package_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "pandas", "geopandas", "shapely", "rasterio", "pyproj", "networkx", "fastapi", "pillow", "pydantic"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "не установлен"
    return out


# --------------------------------------------------------------------------- static demo
def build_demo(out: Path) -> dict:
    from fastapi.testclient import TestClient

    from ecocorridor import api

    client = TestClient(api.app)
    demo = out / "demo"
    data = demo / "data"
    if demo.exists():
        shutil.rmtree(demo)
    shutil.copytree(ROOT / "src/ecocorridor/static", demo)
    data.mkdir(parents=True)

    project = client.get("/api/project").json()
    (data / "project.json").write_text(json.dumps(project, ensure_ascii=False), encoding="utf-8")

    params = project["defaults"]
    tokens = {}
    for sid in SCENARIOS:
        folder = data / sid
        folder.mkdir()
        result = client.post("/api/analyze", json={"scenario": sid, "params": params}).json()
        token = result["token"]
        tokens[sid] = token
        for layer in result["layers"]:
            png = client.get(f"/api/layers/{token}/{layer}.png")
            png.raise_for_status()
            (folder / f"{layer}.png").write_bytes(png.content)
            result["layers"][layer]["url"] = f"data/{sid}/{layer}.png"
        current = client.get(f"/api/layers/{token}/current.png")
        current.raise_for_status()
        (folder / "current.png").write_bytes(current.content)
        summary = client.get(f"/api/circuit/{token}")
        summary.raise_for_status()
        (folder / "circuit.json").write_text(json.dumps(summary.json(), ensure_ascii=False), encoding="utf-8")
        for name, route in (("report.html", "report.html"), ("corridors.geojson", "corridors.geojson"), ("metrics.csv", "metrics.csv")):
            resp = client.get(f"/api/export/{token}/{route}")
            resp.raise_for_status()
            (folder / name).write_bytes(resp.content)
        (folder / "analysis.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    compare = client.post("/api/compare", json={"scenarios": list(SCENARIOS), "params": params}).json()
    (data / "compare.json").write_text(json.dumps(compare, ensure_ascii=False), encoding="utf-8")
    (data / "tokens.json").write_text(json.dumps(tokens), encoding="utf-8")

    # --- API shim: the client talks to files instead of a server
    shim = """/* EcoCorridor — статическая версия: читает заранее рассчитанные результаты */
const API = (() => {
  const base = "data/";
  const getJSON = async (url) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Не удалось загрузить ${url}: ${response.status}`);
    return response.json();
  };
  const tokens = {};
  const off = (what) => Promise.reject(new Error(`${what}: недоступно в статической версии. Запустите приложение локально.`));
  return {
    setProject: () => {},
    projectId: () => "demo",
    project: () => getJSON(base + "project.json"),
    projects: () => Promise.resolve({ projects: [] }),
    analyze: async (scenario) => {
      const result = await getJSON(`${base}${scenario}/analysis.json`);
      tokens[result.token] = scenario;
      return result;
    },
    compare: () => getJSON(base + "compare.json"),
    route: () => off("Прокладка маршрута"),
    sample: () => Promise.reject(new Error("static")),
    currentUrl: (token) => `${base}${tokens[token]}/current.png`,
    circuit: (token) => getJSON(`${base}${tokens[token]}/circuit.json`),
    importData: () => off("Загрузка своих данных"),
    searchPlace: () => off("Поиск территории"),
    planTerritory: () => off("Загрузка территории"),
    loadTerritory: () => off("Загрузка территории"),
    job: () => off("Загрузка территории"),
    reportUrl: (token) => `${base}${tokens[token]}/report.html`,
    geojsonUrl: (token) => `${base}${tokens[token]}/corridors.geojson`,
    csvUrl: (token) => `${base}${tokens[token]}/metrics.csv`,
    gpkgUrl: () => "#",
    rasterUrl: () => "#",
  };
})();
"""
    (demo / "js/api.js").write_text(shim, encoding="utf-8")

    page = (demo / "index.html").read_text(encoding="utf-8")
    page = page.replace('"/static/', '"').replace("'/static/", "'")
    banner = (
        '<div class="static-banner"><b>Статическая версия.</b> Показаны предрассчитанные результаты демонстрационного '
        'полигона при параметрах по умолчанию; ползунки, правки, маршруты, загрузка территорий и своих данных отключены; слой «Ток» (теория цепей) предрассчитан. '
        '<a href="../index.html">← к документации</a> · <a href="../guide.html">справка</a> · '
        '<a href="https://github.com/DmitryLeLeE/eco_corridor">код</a></div>'
    )
    og = (
        '<meta property="og:type" content="website"><meta property="og:title" content="EcoCorridor — демо">'
        '<meta property="og:description" content="Интерактивная демонстрация: сопротивление, коридоры, ток, сценарии">'
        f'<meta property="og:url" content="{SITE_URL}demo/index.html"><meta property="og:image" content="{SITE_URL}img/social.png">'
        '<meta name="twitter:card" content="summary_large_image">'
    )
    page = page.replace("</head>", og + "</head>", 1)
    page = page.replace('<div class="app">', banner + '\n<div class="app">', 1)
    lock = """<script>
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".sidebar input, .sidebar select, [data-reset]").forEach((n) => { n.disabled = true; });
  ["btn-run", "btn-territory", "tool-route", "edits-panel", "class-editor", "btn-gpkg", "btn-raster"].forEach((id) => { const n = document.getElementById(id); if (n) n.style.display = "none"; });
});
</script>
</body>"""
    page = page.replace("</body>", lock, 1)
    (demo / "index.html").write_text(page, encoding="utf-8")
    with (demo / "css/app.css").open("a", encoding="utf-8") as fh:
        fh.write(
            "\n.static-banner{position:relative;z-index:20;padding:7px 16px;font-size:12.5px;background:#fff7e8;"
            "border-bottom:1px solid #e6c98f;color:#5b4300}.static-banner a{margin-left:6px}\n"
            "body{overflow:auto}.app{height:calc(100vh - 34px)}\n"
        )
    return {"tokens": tokens, "project": project["name"], "defaults": params}


# --------------------------------------------------------------------------- documentation
TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — EcoCorridor</title>
<meta name="description" content="{description}">
<meta name="citation_title" content="EcoCorridor: {title}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="EcoCorridor">
<meta property="og:locale" content="ru_RU">
<meta property="og:title" content="{title} — EcoCorridor">
<meta property="og:description" content="{description}">
<meta property="og:url" content="{site}{page_name}.html">
<meta property="og:image" content="{site}img/social.png">
<meta property="og:image:width" content="1280">
<meta property="og:image:height" content="640">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title} — EcoCorridor">
<meta name="twitter:description" content="{description}">
<meta name="twitter:image" content="{site}img/social.png">
<link rel="stylesheet" href="site.css">
</head>
<body>
<header class="top">
  <a class="brand" href="index.html"><span class="mark"></span>EcoCorridor</a>
  <nav>{nav}</nav>
</header>
<div class="wrap">
  <aside class="toc">{toc}</aside>
  <main class="doc">
{body}
  </main>
</div>
<footer>
  <span>EcoCorridor v{version} · сборка {built} · коммит <code>{commit}</code></span>
  <span><a href="https://github.com/DmitryLeLeE/eco_corridor">исходный код</a> · <a href="science.html#cite">как цитировать</a> · <a href="provenance.json">provenance.json</a></span>
</footer>
</body>
</html>
"""

SITE_CSS = """
:root{--text:#17222d;--dim:#3b4957;--muted:#667584;--line:#cfd6dd;--soft:#eef1f4;--accent:#0a3d91;--amber:#9a5b00}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;font:16px/1.65 "Source Sans 3","Segoe UI","Helvetica Neue",Arial,sans-serif;color:var(--text);background:#fff}
h1,h2,h3,h4,.brand{font-family:"Source Serif 4",Georgia,"Times New Roman",serif;line-height:1.25;color:#101a24}
a{color:var(--accent)}
.top{position:sticky;top:0;z-index:5;display:flex;gap:28px;align-items:center;padding:10px 24px;background:#fff;border-bottom:1px solid var(--line)}
.brand{font-size:19px;font-weight:700;text-decoration:none;color:#101a24;display:flex;align-items:center;gap:9px;white-space:nowrap}
.mark{width:18px;height:18px;border:2px solid var(--accent);border-radius:3px;display:inline-block;background:linear-gradient(135deg,transparent 55%,var(--accent) 56%)}
nav{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:14.5px}
nav a{color:var(--dim);text-decoration:none;padding:3px 0;border-bottom:2px solid transparent}
nav a:hover{color:var(--accent)}nav a.on{color:var(--accent);border-bottom-color:var(--accent);font-weight:600}
.wrap{display:grid;grid-template-columns:230px minmax(0,820px);gap:40px;max-width:1160px;margin:0 auto;padding:28px 24px 60px}
.toc{position:sticky;top:64px;align-self:start;max-height:calc(100vh - 90px);overflow:auto;font-size:13.5px;border-right:1px solid var(--soft);padding-right:14px}
.toc ul{list-style:none;margin:0;padding-left:12px}.toc>div>ul{padding-left:0}.toc a{color:var(--dim);text-decoration:none;display:block;padding:2px 0}.toc a:hover{color:var(--accent)}
.doc h1{font-size:32px;margin:0 0 6px}.doc h2{font-size:23px;margin:2.1em 0 .5em;padding-top:.4em;border-top:1px solid var(--soft)}
.doc h3{font-size:18px;margin:1.6em 0 .4em}.doc p,.doc li{max-width:75ch}
.doc table{border-collapse:collapse;margin:1em 0;font-size:14.5px;width:100%;display:block;overflow-x:auto}
.doc th,.doc td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
.doc th{background:var(--soft);font-weight:600}
.doc code{font:13.5px "Cascadia Mono",Consolas,monospace;background:var(--soft);padding:1px 5px;border-radius:3px}
.doc pre{background:#f6f8fa;border:1px solid var(--line);border-radius:4px;padding:12px 14px;overflow:auto}.doc pre code{background:none;padding:0}
.doc blockquote{margin:1em 0;padding:8px 16px;border-left:3px solid var(--amber);background:#fff7e8;color:#4a3600}
.doc img{max-width:100%;border:1px solid var(--line);border-radius:4px}
.doc .lead{font-size:18px;color:var(--dim)}
.doc .eq{margin:1em 0;padding:10px 16px;background:#f6f8fa;border-left:3px solid var(--accent);font:15px "Cambria Math","STIX Two Math","Times New Roman",serif;overflow-x:auto}
.status-ok{color:#1b6b3a;font-weight:600}.status-part{color:var(--amber);font-weight:600}.status-no{color:#a32a2a;font-weight:600}
.cta{display:inline-block;margin:6px 10px 6px 0;padding:9px 18px;border-radius:4px;border:1px solid var(--accent);text-decoration:none;font-weight:600}
.cta.primary{background:var(--accent);color:#fff}
footer{display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;padding:16px 24px;border-top:1px solid var(--line);font-size:12.5px;color:var(--muted)}
@media(max-width:900px){.wrap{grid-template-columns:1fr}.toc{display:none}.top{flex-direction:column;align-items:flex-start;gap:6px}}
@media print{.top,.toc,footer{display:none}.wrap{display:block}}
"""


def sensitivity_table() -> str:
    path = ROOT / "docs/data/sensitivity_demo.json"
    if not path.exists():
        return "_Результаты чувствительности не найдены: запустите `python scripts/sensitivity.py --json docs/data/sensitivity_demo.json`._"
    s = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        "| Показатель | Значение |",
        "|---|---|",
        f"| Проект | {s['project']} |",
        f"| Число прогонов; σ логнормального множителя классов; разброс весов | {s['draws']}; {s['sigma']}; ±{s['weight_spread']:.0%} |",
        f"| Остов сети (MST): Jaccard с базовым, среднее / минимум | {s['backbone_jaccard']['mean']:.2f} / {s['backbone_jaccard']['min']:.2f} |",
        f"| Положение коридоров (IoU полос шириной коридора), среднее / минимум | {s['corridor_cells_iou']['mean']:.2f} / {s['corridor_cells_iou']['min']:.2f} |",
        f"| Стоимость сети, коэффициент вариации | {s['network_cost']['cv']:.0%} |",
        f"| PC в прогонах, среднее ± SD | {s['pc']['mean']:.4f} ± {s['pc']['sd']:.4f} (не зависит от сопротивления по построению) |",
        f"| PC по стоимости в прогонах, среднее ± SD (CV) | {s['pc_cost']['mean']:.4f} ± {s['pc_cost']['sd']:.4f} ({s['pc_cost']['cv']:.1%}) |",
        f"| Порядок сценариев по ΔPC (по стоимости) сохранён | {s['scenario_order_kept']:.0%} прогонов |",
    ]
    freq = ["", "| Связь (пара ядер) | доля прогонов, когда связь входит в остов | входит в базовый остов |", "|---|---|---|"]
    base_backbone = {tuple(x) for x in s["baseline"]["backbone"]}
    for key, share in s["backbone_frequency"].items():
        a, b = (int(x) for x in key.split("-"))
        freq.append(f"| {key} | {share:.0%} | {'да' if (a, b) in base_backbone else 'нет'} |")
    return "\n".join(rows + freq)


def build_docs(out: Path, meta: dict) -> None:
    built = meta["built"]
    nav_items = [(n, t) for n, (_s, t) in PAGES.items()]

    def nav(current: str) -> str:
        links = [f'<a href="{n}.html"{" class=on" if n == current else ""}>{t}</a>' for n, t in nav_items]
        links.insert(1, '<a href="demo/index.html">Демо</a>')
        return "".join(links)

    for name, (src, title) in PAGES.items():
        text = (ROOT / src).read_text(encoding="utf-8")
        text = text.replace("{{SENSITIVITY_TABLE}}", sensitivity_table()).replace("{{VERSION}}", __version__)
        text = re.sub(r"\]\(([\w./-]*?)([\w-]+\.md)(#[\w-]+)?\)", lambda m: f"]({LINKS.get(m.group(2), m.group(2))}{m.group(3) or ''})", text)
        text = text.replace("(docs/img/", "(img/").replace("](img/", "](img/")
        md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "attr_list", "footnotes", "sane_lists", "md_in_html"], extension_configs={"toc": {"toc_depth": "2-3", "title": ""}})
        body = md.convert(text)
        first = re.search(r"<p>(.*?)</p>", body, re.S)
        description = html.escape(re.sub(r"<[^>]+>", "", first.group(1))[:200]) if first else "EcoCorridor"
        page = TEMPLATE.format(site=SITE_URL, page_name=name, title=html.escape(title), description=description, nav=nav(name), toc=md.toc, body=body, version=__version__, built=built, commit=meta["commit"][:7] or "—")
        (out / f"{name}.html").write_text(page, encoding="utf-8")
    (out / "site.css").write_text(SITE_CSS, encoding="utf-8")
    shutil.copytree(ROOT / "docs/img", out / "img", dirs_exist_ok=True)
    (out / ".nojekyll").write_text("", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta = {"built": built, "commit": git("rev-parse", "HEAD"), "version": __version__}
    demo = build_demo(out)
    build_docs(out, meta)
    provenance = {
        **meta,
        "python_packages": package_versions(),
        "platform": platform.platform(),
        "demo": {"project": demo["project"], "scenario_tokens": demo["tokens"], "parameters": demo["defaults"]},
        "note": "Токен сценария — SHA-1 от (проект, сценарий, параметры); одинаковые параметры дают одинаковый токен.",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"site built in {out} ({sum(1 for _ in out.rglob('*') if _.is_file())} files)")


if __name__ == "__main__":
    main()
