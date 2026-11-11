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

  return {
    project: () => request("/api/project"),
    analyze: (scenario, params) => request("/api/analyze", { method: "POST", body: JSON.stringify({ scenario, params }) }),
    compare: (scenarios, params) => request("/api/compare", { method: "POST", body: JSON.stringify({ scenarios, params }) }),
    route: (token, start, end) => request("/api/route", { method: "POST", body: JSON.stringify({ token, start, end }) }),
    sample: (token, x, y) => request(`/api/sample/${token}?x=${x.toFixed(1)}&y=${y.toFixed(1)}`),
    reportUrl: (token) => `/api/export/${token}/report.html`,
    geojsonUrl: (token) => `/api/export/${token}/corridors.geojson`,
    csvUrl: (token) => `/api/export/${token}/metrics.csv`,
  };
})();
