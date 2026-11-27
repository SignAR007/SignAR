const MAX_BODY_BYTES = 128 * 1024;
const MAX_ENTRIES_PER_REQUEST = 50;

function response(request, env, body, status = 200) {
  const origin = request.headers.get("Origin");
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Access-Control-Allow-Headers": "Content-Type",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Origin": origin,
      "Access-Control-Max-Age": "86400",
      "Cache-Control": "no-store",
      "Content-Type": "application/json; charset=utf-8",
      Vary: "Origin",
    },
  });
}

function isAllowedOrigin(request, env) {
  const origin = request.headers.get("Origin");
  if (!origin) return false;
  return (env.ALLOWED_ORIGINS || "")
    .split(",")
    .map((allowed) => allowed.trim())
    .includes(origin);
}

function normalizeEntries(entries) {
  if (
    !Array.isArray(entries) ||
    entries.length < 1 ||
    entries.length > MAX_ENTRIES_PER_REQUEST
  )
    return null;

  const normalized = [];
  for (const entry of entries) {
    if (!entry || typeof entry !== "object") return null;
    const alias = typeof entry.alias === "string" ? entry.alias.trim() : "";
    const message =
      typeof entry.message === "string" ? entry.message.trim() : "";
    const id =
      typeof entry.id === "string" && /^[A-Za-z0-9._:-]{1,128}$/.test(entry.id)
        ? entry.id
        : crypto.randomUUID();
    const createdAt =
      typeof entry.createdAt === "string" &&
      !Number.isNaN(Date.parse(entry.createdAt))
        ? new Date(entry.createdAt).toISOString()
        : new Date().toISOString();

    if (alias.length > 80 || message.length < 1 || message.length > 2000)
      return null;
    normalized.push({ id, alias, message, createdAt });
  }
  return normalized;
}

export default {
  async fetch(request, env) {
    if (!isAllowedOrigin(request, env))
      return new Response("Origin not allowed", { status: 403 });

    if (request.method === "OPTIONS")
      return new Response(null, {
        status: 204,
        headers: {
          "Access-Control-Allow-Headers": "Content-Type",
          "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
          "Access-Control-Allow-Origin": request.headers.get("Origin"),
          "Access-Control-Max-Age": "86400",
          Vary: "Origin",
        },
      });

    const url = new URL(request.url);
    if (url.pathname !== "/api/feedback")
      return response(request, env, { error: "Not found" }, 404);

    try {
      if (request.method === "GET") {
        const result = await env.DB.prepare(
          "SELECT id, alias, message, created_at AS createdAt FROM feedback ORDER BY created_at DESC",
        ).all();
        return response(request, env, { entries: result.results });
      }

      if (request.method !== "POST")
        return response(request, env, { error: "Method not allowed" }, 405);

      const contentLength = Number(request.headers.get("Content-Length") || 0);
      if (contentLength > MAX_BODY_BYTES)
        return response(request, env, { error: "Request too large" }, 413);
      const rawBody = await request.text();
      if (new TextEncoder().encode(rawBody).byteLength > MAX_BODY_BYTES)
        return response(request, env, { error: "Request too large" }, 413);

      let payload;
      try {
        payload = JSON.parse(rawBody);
      } catch {
        return response(request, env, { error: "Invalid JSON" }, 400);
      }

      const entries = normalizeEntries(payload?.entries);
      if (!entries)
        return response(
          request,
          env,
          { error: "Invalid feedback entries" },
          400,
        );

      const statements = entries.map((entry) =>
        env.DB.prepare(
          "INSERT OR IGNORE INTO feedback (id, alias, message, created_at) VALUES (?, ?, ?, ?)",
        ).bind(entry.id, entry.alias, entry.message, entry.createdAt),
      );
      const results = await env.DB.batch(statements);
      const added = results.reduce(
        (total, result) => total + Number(result.meta?.changes || 0),
        0,
      );
      return response(request, env, { added }, added ? 201 : 200);
    } catch (error) {
      console.error("Feedback API request failed.", error);
      return response(
        request,
        env,
        { error: "Feedback service unavailable" },
        503,
      );
    }
  },
};
