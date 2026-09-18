import { checkAbort, DecisionError } from "./errors.js";

export type CacheMode = "auto" | "none" | "refresh";
export interface LoadProgress {
  stage:
    | "manifest"
    | "tokenizer"
    | "model"
    | "initializing"
    | "ready"
    | "warning";
  message: string;
  file?: string;
  loaded?: number;
  total?: number;
  /** Fraction from 0 to 1 for the current file, not the entire model. */
  progress?: number;
  cached?: boolean;
}
export interface AssetOptions {
  signal?: AbortSignal;
  cache?: CacheMode;
  onProgress?: (event: LoadProgress) => void;
}
export interface ModelManifest {
  format: number;
  precision: "fp16" | "fp32";
  temperature: number;
  max_length: number;
  pad_token_id: number;
  state_names: string[];
  state_shapes: number[][];
  max_choices?: number;
  model_name?: "decision-lab-v2" | "decision-lab-v3";
  weights_bytes: number;
  graph_bytes: number;
  checkpoint_sha256?: string;
  [key: string]: unknown;
}

export function normalizeModelUrl(
  value: string,
  origin = globalThis.location?.href,
): string {
  let url: URL;
  try {
    url = new URL(value, origin);
  } catch {
    throw new DecisionError(
      "INVALID_MODEL",
      "Provide an absolute model URL or a browser-relative directory.",
    );
  }
  if (
    !["https:", "http:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  )
    throw new DecisionError(
      "INVALID_MODEL",
      "Use an HTTP(S) model directory without credentials, query, or fragment.",
    );
  return url.href.replace(/\/+$/, "") + "/";
}

export function parseManifest(value: unknown): ModelManifest {
  const m = value as ModelManifest;
  if (
    !m ||
    m.format !== 1 ||
    !["fp16", "fp32"].includes(m.precision) ||
    !Number.isFinite(m.temperature) ||
    m.temperature <= 0 ||
    !Number.isInteger(m.max_length) ||
    m.max_length < 8 ||
    m.max_length > 32768 ||
    !Number.isInteger(m.pad_token_id) ||
    m.pad_token_id < 0 ||
    !Number.isInteger(m.weights_bytes) ||
    m.weights_bytes < 1 ||
    m.weights_bytes > 2_000_000_000 ||
    !Number.isInteger(m.graph_bytes) ||
    m.graph_bytes < 1 ||
    m.graph_bytes > 100_000_000 ||
    ![4, 8].includes(m.max_choices ?? 4) ||
    !["decision-lab-v2", "decision-lab-v3"].includes(
      m.model_name ?? "decision-lab-v2",
    ) ||
    !Array.isArray(m.state_names) ||
    !m.state_names.length ||
    m.state_names.length > 128 ||
    m.state_names.some(
      (n) => typeof n !== "string" || !/^(conv|key|value)_\d+$/.test(n),
    ) ||
    new Set(m.state_names).size !== m.state_names.length ||
    !Array.isArray(m.state_shapes) ||
    m.state_shapes.length !== m.state_names.length ||
    m.state_shapes.some(
      (shape) =>
        !Array.isArray(shape) ||
        !shape.length ||
        shape.length > 5 ||
        shape.some((n) => !Number.isInteger(n) || n < 1) ||
        shape.reduce((a, b) => a * b, 1) > 10_000_000,
    )
  )
    throw new DecisionError(
      "INVALID_MODEL",
      "Invalid or unsupported Decision Lab model manifest.",
    );
  return m;
}

const cachePrefix = "decision-lab-sdk-v1:";
async function digest(text: string): Promise<string> {
  return Array.from(
    new Uint8Array(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)),
    ),
    (n) => n.toString(16).padStart(2, "0"),
  ).join("");
}

/** Deletes only this SDK's cached artifacts for this directory, never a loaded model. */
export async function clearModelCache(modelUrl: string): Promise<number> {
  if (!globalThis.caches) return 0;
  const prefix =
    cachePrefix + (await digest(normalizeModelUrl(modelUrl))) + ":";
  const names = (await caches.keys()).filter((name) => name.startsWith(prefix));
  return (await Promise.all(names.map((name) => caches.delete(name)))).filter(
    Boolean,
  ).length;
}

export class ModelAssets {
  private cache?: Cache;
  constructor(
    readonly base: string,
    private readonly options: AssetOptions = {},
  ) {}

  emit(event: LoadProgress): void {
    this.options.onProgress?.(event);
  }

  async manifest(): Promise<ModelManifest> {
    // Revalidate the small manifest; immutable, versioned directories are recommended.
    const bytes = await this.read(
      "manifest.json",
      "manifest",
      undefined,
      false,
    );
    let manifest: ModelManifest;
    const text = new TextDecoder().decode(bytes);
    try {
      manifest = parseManifest(JSON.parse(text));
    } catch (cause) {
      throw new DecisionError(
        "INVALID_MODEL",
        "Cannot read the model manifest.",
        { cause },
      );
    }
    if (this.options.cache !== "none" && globalThis.caches) {
      try {
        this.cache = await caches.open(
          cachePrefix + (await digest(this.base)) + ":" + (await digest(text)),
        );
      } catch {
        this.emit({
          stage: "warning",
          message:
            "Browser storage unavailable; continuing without persistent cache.",
        });
      }
    }
    return manifest;
  }

  async read(
    file: string,
    stage: "manifest" | "tokenizer" | "model",
    expected?: number,
    cacheable = true,
  ): Promise<Uint8Array> {
    checkAbort(this.options.signal);
    const url = this.base + file;
    let response: Response | undefined;
    let cached = false;
    if (cacheable && this.options.cache !== "refresh" && this.cache) {
      try {
        response = await this.cache.match(url);
        cached = !!response;
      } catch {
        this.emit({
          stage: "warning",
          message: "Cache read failed; downloading the artifact.",
        });
      }
    }
    if (!response) {
      try {
        response = await fetch(url, {
          signal: this.options.signal,
          cache: "no-store",
        });
      } catch (cause) {
        checkAbort(this.options.signal);
        throw new DecisionError(
          "DOWNLOAD_FAILED",
          `Could not download ${file}.`,
          { cause },
        );
      }
    }
    if (!response.ok)
      throw new DecisionError(
        "DOWNLOAD_FAILED",
        `${file}: HTTP ${response.status}.`,
      );
    const length = Number(response.headers.get("content-length"));
    const total =
      expected ??
      (Number.isSafeInteger(length) && length > 0 ? length : undefined);
    const limit = expected ?? (stage === "manifest" ? 1_000_000 : 50_000_000);
    // Allocate large, known-size model files once rather than duplicating them on concatenation.
    const output = expected ? new Uint8Array(expected) : undefined;
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    const report = () =>
      this.emit({
        stage,
        file,
        loaded,
        total,
        progress: total ? Math.min(loaded / total, 1) : undefined,
        cached,
        message: `${cached ? "Reading cached" : "Downloading"} ${file}`,
      });
    report();
    const reader = response.body?.getReader();
    if (!reader)
      throw new DecisionError(
        "DOWNLOAD_FAILED",
        `${file}: response has no body.`,
      );
    try {
      while (true) {
        checkAbort(this.options.signal);
        const { done, value } = await reader.read();
        if (done) break;
        loaded += value.byteLength;
        if (loaded > limit)
          throw new DecisionError(
            "INVALID_MODEL",
            `${file}: artifact exceeds its declared size limit.`,
          );
        if (output) output.set(value, loaded - value.byteLength);
        else chunks.push(value);
        report();
      }
    } catch (error) {
      await reader.cancel().catch(() => {});
      throw error;
    } finally {
      reader.releaseLock();
    }
    checkAbort(this.options.signal);
    if (expected && loaded !== expected) {
      if (cached) await this.cache?.delete(url).catch(() => {});
      throw new DecisionError(
        "INVALID_MODEL",
        `${file}: expected ${expected} bytes, received ${loaded}. Retry with cache: 'refresh'.`,
      );
    }
    const bytes = output ?? new Uint8Array(loaded);
    if (!output) {
      let offset = 0;
      for (const chunk of chunks) {
        bytes.set(chunk, offset);
        offset += chunk.byteLength;
      }
    }
    if (!cached && cacheable && this.cache) {
      try {
        await this.cache.put(
          url,
          new Response(bytes as Uint8Array<ArrayBuffer>, {
            headers: { "content-length": String(loaded) },
          }),
        );
      } catch {
        this.emit({
          stage: "warning",
          message:
            "Could not save this artifact to browser storage; inference can continue.",
        });
      }
    }
    checkAbort(this.options.signal);
    return bytes;
  }
}
