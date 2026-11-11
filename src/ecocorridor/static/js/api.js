/* EcoCorridor — HTTP client */

const API = (() => {
  async function request(url, options = {}) {
    const response = await fetch(url, {
      headers: options.body ? { "Content-Type": "application/json" } : undefined,
      ...options,
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const payload = await response.json();
        detail = payload.detail || detail;
      } catch (_) { /* not JSON */ }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    return response.headers.get("content-type")?.includes("application/json")
      ? response.json()
      : response.text();
  }

  const post = (url, body) => request(url, { method: "POST", body: JSON.stringify(body) });
  let projectId = null;

  return {
    setProject: (id) => { projectId = id; },
    projectId: () => projectId,
    project: (id) => request("/api/project" + (id ? `?project_id=${encodeURIComponent(id)}` : "")),
    projects: () => request("/api/projects"),
    analyze: (scenario, params) => post("/api/analyze", { project_id: projectId, scenario, params }),
    compare: (scenarios, params) => post("/api/compare", { project_id: projectId, scenarios, params }),
    searchPlace: (q) => request(`/api/territory/search?q=${encodeURIComponent(q)}`),
    planTerritory: (body) => post("/api/territory/plan", body),
    loadTerritory: (body) => post("/api/territory/load", body),
    job: (id) => request(`/api/territory/jobs/${encodeURIComponent(id)}`),
    route: (token, start, end) => request("/api/route", { method: "POST", body: JSON.stringify({ token, start, end }) }),
    sample: (token, x, y) => request(`/api/sample/${token}?x=${x.toFixed(1)}&y=${y.toFixed(1)}`),
    reportUrl: (token) => `/api/export/${token}/report.html`,
    geojsonUrl: (token) => `/api/export/${token}/corridors.geojson`,
    csvUrl: (token) => `/api/export/${token}/metrics.csv`,
  };
})();
