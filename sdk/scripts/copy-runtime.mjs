#!/usr/bin/env node
import { copyFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve, join } from "node:path";

const directory = resolve(process.argv[2] ?? "public/ort");
await mkdir(directory, { recursive: true });
for (const name of [
  "ort-wasm-simd-threaded.asyncify.mjs",
  "ort-wasm-simd-threaded.asyncify.wasm",
]) {
  await copyFile(
    fileURLToPath(import.meta.resolve(`onnxruntime-web/${name}`)),
    join(directory, name),
  );
}
console.log(`Copied ONNX Runtime 1.24.3 support files to ${directory}`);
