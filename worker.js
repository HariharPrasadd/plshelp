const TRACKED_PATHS = new Set(["/install.sh", "/install.ps1"]);

function classifyUserAgent(userAgent) {
  const ua = (userAgent || "").toLowerCase();

  if (!ua) {
    return "unknown";
  }
  if (ua.includes("powershell")) {
    return "powershell";
  }
  if (ua.includes("curl")) {
    return "curl";
  }
  if (ua.includes("wget")) {
    return "wget";
  }
  if (ua.includes("python-requests")) {
    return "python-requests";
  }
  if (ua.includes("go-http-client")) {
    return "go-http-client";
  }

  return "other";
}

function trackDownload(request, env, response) {
  if (!env.DOWNLOADS) {
    return;
  }

  const url = new URL(request.url);
  const country = request.cf?.country || "unknown";
  const colo = request.cf?.colo || "unknown";
  const asn = Number.isFinite(request.cf?.asn) ? request.cf.asn : 0;
  const status = response?.status || 0;
  const method = request.method || "GET";
  const uaClass = classifyUserAgent(request.headers.get("user-agent"));

  env.DOWNLOADS.writeDataPoint({
    blobs: [url.pathname, method, uaClass, country, colo],
    doubles: [1, status, asn],
    indexes: [url.hostname],
  });
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const response = await env.ASSETS.fetch(request);

    if (TRACKED_PATHS.has(url.pathname)) {
      ctx.waitUntil(Promise.resolve(trackDownload(request, env, response)));
    }

    return response;
  },
};
