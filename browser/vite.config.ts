import { defineConfig } from "vite";
export default defineConfig({
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  resolve: { dedupe: ["onnxruntime-web", "@huggingface/transformers", "ajv"] },
  optimizeDeps: {
    exclude: ["onnxruntime-web", "onnxruntime-web/webgpu", "decision-lab-sdk"],
  },
  build: { rollupOptions: { input: { main: "index.html", sdk: "sdk.html" } } },
});
