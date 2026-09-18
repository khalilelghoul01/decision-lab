import { DecisionModel } from "./client.js";
import { checkAbort, DecisionError } from "./errors.js";
import type { AssetOptions } from "./assets.js";

export { DecisionModel, choice, noul, score } from "./client.js";
export type {
  DecisionRequest,
  DecisionResponse,
  Questions,
  InferenceOptions,
} from "./client.js";
export { DecisionError } from "./errors.js";
export type { ErrorCode } from "./errors.js";
export { clearModelCache } from "./assets.js";
export type { LoadProgress, CacheMode, ModelManifest } from "./assets.js";
export type {
  Description,
  JevQuestion,
  JevAnswer,
  JevRequest,
  JevResponse,
} from "./jev-protocol.js";

export interface LoadModelOptions extends AssetOptions {
  /** URL of an exported model directory. Default: /model-v3-q4. */
  modelUrl?: string;
  /** Same-version ONNX Runtime module and binary URLs; self-host both for offline use. */
  wasmPaths?: { mjs: string; wasm: string };
  fieldBatchSize?: number;
}

/** Load once, reuse for many requests, then await model.dispose(). Browser WebGPU only. */
export async function loadModel(
  options: LoadModelOptions = {},
): Promise<DecisionModel> {
  checkAbort(options.signal);
  if (!globalThis.navigator || !("gpu" in globalThis.navigator))
    throw new DecisionError(
      "UNSUPPORTED_ENVIRONMENT",
      "Decision Lab needs browser WebGPU on localhost or HTTPS. Node/server inference is not included.",
    );
  const { BrowserDecisionEngine } = await import("./engine.js");
  checkAbort(options.signal);
  const wasmPaths = options.wasmPaths ?? {
    mjs: new URL(
      "/ort/ort-wasm-simd-threaded.asyncify.mjs",
      globalThis.location.href,
    ).href,
    wasm: new URL(
      "/ort/ort-wasm-simd-threaded.asyncify.wasm",
      globalThis.location.href,
    ).href,
  };
  const runtime = await BrowserDecisionEngine.load(
    options.modelUrl ?? "/model-v3-q4",
    () => {},
    "webgpu",
    { ...options, wasmPaths, cache: options.cache ?? "auto" },
  );
  return new DecisionModel(runtime);
}
