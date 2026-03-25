const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "host",
  "content-length",
]);

function normalizeBackendBase(rawBase) {
  if (typeof rawBase !== "string") {
    return "";
  }
  return rawBase.trim().replace(/\/+$/, "");
}

function readRequestBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];

    req.on("data", (chunk) => {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    });

    req.on("end", () => {
      if (chunks.length === 0) {
        resolve(undefined);
        return;
      }
      resolve(Buffer.concat(chunks));
    });

    req.on("error", reject);
  });
}

function buildTargetUrl(req, backendBase) {
  const pathValue = req.query?.path;
  const pathSegments = Array.isArray(pathValue) ? pathValue : pathValue ? [pathValue] : [];
  const sanitizedPath = pathSegments
    .map((segment) => String(segment).replace(/^\/+|\/+$/g, ""))
    .filter(Boolean)
    .join("/");

  const target = new URL(`${backendBase}/${sanitizedPath}`);
  for (const [key, value] of Object.entries(req.query || {})) {
    if (key === "path" || value === undefined) {
      continue;
    }
    if (Array.isArray(value)) {
      value.forEach((item) => target.searchParams.append(key, String(item)));
      continue;
    }
    target.searchParams.append(key, String(value));
  }
  return target;
}

function copyRequestHeaders(req) {
  const headers = new Headers();
  for (const [key, value] of Object.entries(req.headers || {})) {
    if (!value) {
      continue;
    }
    const normalizedKey = key.toLowerCase();
    if (HOP_BY_HOP_HEADERS.has(normalizedKey)) {
      continue;
    }
    headers.set(key, Array.isArray(value) ? value.join(", ") : String(value));
  }
  return headers;
}

module.exports = async function handler(req, res) {
  const backendBase = normalizeBackendBase(process.env.BACKEND_API_URL);
  if (!backendBase) {
    res.statusCode = 503;
    res.setHeader("content-type", "application/json");
    res.end(
      JSON.stringify({
        detail:
          "BACKEND_API_URL is not configured for the Vercel proxy. Set it to your FastAPI backend base URL.",
      })
    );
    return;
  }

  const targetUrl = buildTargetUrl(req, backendBase);
  const headers = copyRequestHeaders(req);

  let body;
  if (req.method !== "GET" && req.method !== "HEAD") {
    body = await readRequestBody(req);
    if (body && body.length > 0) {
      headers.set("content-length", String(body.length));
    }
  }

  let upstream;
  try {
    upstream = await fetch(targetUrl, {
      method: req.method,
      headers,
      body,
      redirect: "manual",
    });
  } catch (error) {
    res.statusCode = 502;
    res.setHeader("content-type", "application/json");
    res.end(
      JSON.stringify({
        detail: `Failed to reach backend API proxy target: ${error instanceof Error ? error.message : "Unknown error"}`,
      })
    );
    return;
  }

  res.statusCode = upstream.status;
  upstream.headers.forEach((value, key) => {
    if (HOP_BY_HOP_HEADERS.has(key.toLowerCase())) {
      return;
    }
    res.setHeader(key, value);
  });

  const payload = Buffer.from(await upstream.arrayBuffer());
  res.end(payload);
};
