import { test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  ModelAssets,
  normalizeModelUrl,
  parseManifest,
  clearModelCache,
} from "../dist/assets.js";
const manifest = JSON.parse(
  await readFile(new URL("./fixtures/manifest.json", import.meta.url), "utf8"),
);

test("normalizes local and nested model URLs; rejects misleading URLs", () => {
  assert.equal(
    normalizeModelUrl("./models/q4", "https://example.com/app/"),
    "https://example.com/app/models/q4/",
  );
  assert.equal(
    normalizeModelUrl("https://example.com/models/q4///"),
    "https://example.com/models/q4/",
  );
  for (const url of [
    "file:///tmp/model",
    "https://user:password@example.com/model",
    "https://example.com/model?token=x",
  ])
    assert.throws(() => normalizeModelUrl(url));
});
test("validates the actual manifest and rejects invalid cache dimensions", () => {
  assert.equal(parseManifest(manifest).max_choices, 8);
  for (const override of [
    { temperature: 0 },
    { state_shapes: [[1, 1e12]] },
    { state_names: ["bad"] },
    { weights_bytes: -1 },
    { format: 100 },
  ])
    assert.throws(() => parseManifest({ ...manifest, ...override }));
});
test("loader checks HTTP errors, declared sizes, progress and cancellation", async () => {
  const original = globalThis.fetch;
  const events = [];
  try {
    globalThis.fetch = async () => new Response("bad", { status: 404 });
    await assert.rejects(
      new ModelAssets("https://example.com/", { cache: "none" }).read(
        "missing",
        "model",
        3,
      ),
      (e) => e.code === "DOWNLOAD_FAILED",
    );
    globalThis.fetch = async () => new Response(new Uint8Array([1, 2, 3]));
    const loader = new ModelAssets("https://example.com/", {
      cache: "none",
      onProgress: (e) => events.push(e),
    });
    assert.deepEqual([...(await loader.read("file", "model", 3))], [1, 2, 3]);
    assert.equal(events.at(-1).progress, 1);
    await assert.rejects(
      loader.read("file", "model", 4),
      (e) => e.code === "INVALID_MODEL",
    );
    await assert.rejects(
      loader.read("file", "model", 2),
      (e) => e.code === "INVALID_MODEL",
    );
    const aborted = new AbortController();
    aborted.abort();
    await assert.rejects(
      new ModelAssets("https://example.com/", { signal: aborted.signal }).read(
        "file",
        "model",
        3,
      ),
      (e) => e.name === "AbortError",
    );
  } finally {
    globalThis.fetch = original;
  }
});
test("cache is reused, refresh replaces it, and clearing is scoped to a model URL", async () => {
  const fetchOriginal = globalThis.fetch,
    cacheOriginal = globalThis.caches;
  const stores = new Map();
  let downloads = 0;
  globalThis.caches = {
    async open(name) {
      if (!stores.has(name)) stores.set(name, new Map());
      const store = stores.get(name);
      return {
        async match(k) {
          return store.get(k)?.clone();
        },
        async put(k, r) {
          store.set(k, r.clone());
        },
        async delete(k) {
          return store.delete(k);
        },
      };
    },
    async keys() {
      return [...stores.keys()];
    },
    async delete(name) {
      return stores.delete(name);
    },
  };
  globalThis.fetch = async (url) => {
    if (url.endsWith("manifest.json")) return Response.json(manifest);
    downloads++;
    return new Response(new Uint8Array([1, 2, 3]));
  };
  try {
    for (const mode of ["auto", "auto", "refresh"]) {
      const a = new ModelAssets("https://example.com/model/", { cache: mode });
      await a.manifest();
      await a.read("weights", "model", 3);
    }
    assert.equal(downloads, 2);
    assert.equal(await clearModelCache("https://example.com/other/"), 0);
    assert.equal(await clearModelCache("https://example.com/model/"), 1);
    assert.equal(stores.size, 0);
  } finally {
    globalThis.fetch = fetchOriginal;
    if (cacheOriginal === undefined) delete globalThis.caches;
    else globalThis.caches = cacheOriginal;
  }
});
