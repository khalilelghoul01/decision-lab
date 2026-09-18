// Compatibility entry for the original demo; the SDK owns the inference code.
import {
  BrowserDecisionEngine as Engine,
  type EngineLoadOptions,
} from "decision-lab-sdk/engine";
import wasmModuleUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import wasmBinaryUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
export { compileSchema } from "decision-lab-sdk/engine";
export type { Prepared, Schema, Value, Row } from "decision-lab-sdk/engine";
export class BrowserDecisionEngine extends Engine {
  static override load(
    base = "/model-v3-q4",
    progress: (message: string) => void = () => {},
    backend: "webgpu" | "wasm" = "webgpu",
    options: EngineLoadOptions = {},
  ) {
    return Engine.load(base, progress, backend, {
      ...options,
      wasmPaths: options.wasmPaths ?? {
        mjs: new URL(wasmModuleUrl, location.href).href,
        wasm: new URL(wasmBinaryUrl, location.href).href,
      },
    });
  }
}
