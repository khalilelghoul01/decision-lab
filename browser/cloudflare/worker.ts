interface Env {
  ASSETS: { fetch(request: Request): Promise<Response> };
  MODELS: {
    head(key: string): Promise<any>;
    get(key: string, options?: { range?: Headers }): Promise<any>;
  };
}

const r2Prefixes = ["/model-v3-q4/", "/assets/ort-wasm-simd-threaded.asyncify-"];
const isR2Asset = (pathname: string) =>
  r2Prefixes.some((prefix) => pathname.startsWith(prefix)) &&
  !pathname.includes("..") &&
  !pathname.endsWith("/");

function assetHeaders(headers: Headers, pathname: string) {
  headers.set("Access-Control-Allow-Origin", "*");
  headers.set("Cross-Origin-Resource-Policy", "cross-origin");
  headers.set("Accept-Ranges", "bytes");
  headers.set("X-Content-Type-Options", "nosniff");
  headers.set("Cache-Control", pathname.includes("decision.weights") || pathname.endsWith(".wasm")
    ? "public, max-age=31536000, immutable"
    : "public, max-age=3600");
}

async function fromR2(request: Request, env: Env, pathname: string) {
  if (request.method !== "GET" && request.method !== "HEAD")
    return new Response("Method not allowed", { status: 405, headers: { Allow: "GET, HEAD" } });
  const key = pathname.slice(1);
  const object = request.method === "HEAD"
    ? await env.MODELS.head(key)
    : await env.MODELS.get(key, { range: request.headers });
  if (!object) return new Response("Asset not found", { status: 404 });
  const headers = new Headers();
  object.writeHttpMetadata?.(headers);
  headers.set("etag", object.httpEtag);
  const range = object.range;
  const isRangeResponse = request.method === "GET" && request.headers.has("range");
  if (isRangeResponse && range && typeof range.offset === "number" && typeof range.length === "number") {
    headers.set("Content-Range", `bytes ${range.offset}-${range.offset + range.length - 1}/${object.size}`);
    headers.set("Content-Length", String(range.length));
  } else headers.set("Content-Length", String(object.size));
  assetHeaders(headers, pathname);
  return new Response(request.method === "HEAD" ? null : object.body, {
    status: isRangeResponse ? 206 : 200,
    headers,
  });
}

export default {
  async fetch(request: Request, env: Env) {
    const url = new URL(request.url);
    if (isR2Asset(url.pathname)) return fromR2(request, env, url.pathname);
    const response = await env.ASSETS.fetch(request);
    const headers = new Headers(response.headers);
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    headers.set("X-Content-Type-Options", "nosniff");
    headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  },
};
