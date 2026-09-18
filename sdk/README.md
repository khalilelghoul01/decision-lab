# Decision Lab SDK

A small TypeScript/JavaScript package for loading the trained Decision Lab model
and making typed decisions locally in a WebGPU browser. It includes a model loader,
download progress, persistent artifact caching, request queuing and GPU cleanup.

The npm package contains code and type declarations. **Model weights are separate**
(about 296 MB for V3 INT4). The package is a GitHub release download; it has not been published to the npm registry.

## Install the local package

```sh
npm install https://github.com/khalilelghoul01/decision-lab/releases/download/v0.1.0/decision-lab-sdk-0.1.0.tgz
npx decision-lab-assets public/ort
```

Copy the entire exported `model-v3-q4` directory into your app's `public` directory.
Use `python3 scripts/download_model.py` from the repository root, or extract
`decision-lab-v3-int4.zip` from GitHub Releases. Preserve its license files.
Serve the app on localhost or HTTPS. The model directory must include:

```text
public/
  ort/                         # copied by decision-lab-assets
  model-v3-q4/
    manifest.json
    decision.onnx
    decision.weights
    tokenizer/
      tokenizer.json
      tokenizer_config.json
      chat_template.jinja
```

## Load once and reuse

```ts
import {loadModel, choice, noul, score} from 'decision-lab-sdk';

const model = await loadModel({
  modelUrl: '/model-v3-q4',
  onProgress(event) {
    // progress is 0–1 for the current file, not the entire download.
    console.log(event.stage, event.file, event.progress, event.cached);
  },
});

const result = await model.systemOne({
  state: 'The plot was awful, but the acting was excellent.',
  questions: {
    good_acting: noul('Was the acting good?'),
    topic: choice('What is this text about?', {
      Movies: null,
      Sports: null,
      Politics: null,
    }),
    acting_quality: score('How good was the acting?', ['Bad', 'Mixed', 'Good']),
  },
}, {mode: 'accurate'});

console.log(result.answers.good_acting.noul); // number: P(yes)
console.log(result.answers.topic.choice);    // typed: Movies | Sports | Politics
console.log(result.answers.acting_quality.score); // expected level index, 0–2

await model.dispose(); // when your app is done with the model
```

Plain JavaScript uses the same API. Literal Jev-shaped request objects also work;
the helper functions are optional. `model` in the request defaults to the loaded
model. `jev-latest` is accepted as a compatibility alias for that local model; it
never calls Jev. Explicitly naming a different loaded-model version is rejected.

## API

| Function / option | Behavior |
| --- | --- |
| `loadModel(options)` | Loads model files, tokenizer and one WebGPU session |
| `model.systemOne(request, options?)` | Returns typed `answers`, `model` and `usage` |
| `mode: 'accurate'` | Default; averages original and reversed choice orders |
| `mode: 'fast'` | Scores one choice order |
| `model.metadata` | Copy of the exported model manifest |
| `model.status` | `ready`, `disposing` or `disposed` |
| `model.dispose()` | Drains accepted requests and releases the session; idempotent |
| `clearModelCache(modelUrl)` | Clears only SDK caches for that model URL |
| `cache: 'auto'` | Default; reuse browser Cache Storage when available |
| `cache: 'none'` | Download without reading/writing SDK persistent caches |
| `cache: 'refresh'` | Download and replace cached artifacts |
| `fieldBatchSize` | 1–32; default 8, reduce to lower peak inference memory |

Concurrent calls to one model are queued. Independent fields within each request
reuse the common context cache. A failed request does not break the queue.
Do not load a new model for every request. Multiple loaded model instances consume
additional GPU memory.

## Cancellation and errors

```ts
const controller = new AbortController();
const loading = loadModel({signal: controller.signal});
// In a cancel button handler: controller.abort();
const model = await loading;

try {
  await model.systemOne(request, {signal: requestController.signal});
} catch (error) {
  if (error instanceof DOMException && error.name === 'AbortError') {
    // Cancelled.
  } else {
    // DecisionError includes a stable `code` plus a human-readable message.
    console.error(error);
  }
}
```

Downloads honor `AbortSignal`. An ONNX session initialization or GPU dispatch
already in progress cannot be preempted: it finishes, then the SDK releases the
new session or discards the cancelled prediction. Queued cancelled requests do
not run inference. Call `dispose()` to release a loaded model.

Errors use `DecisionError.code`: `UNSUPPORTED_ENVIRONMENT`, `DOWNLOAD_FAILED`,
`INVALID_MODEL`, `INVALID_REQUEST`, `INFERENCE_FAILED`, or `DISPOSED`.
Cancellation retains the signal's reason (normally an `AbortError`).

## Hosting and caching

Runtime files default to `/ort/`; the copy command installs the exact files for
the pinned ONNX Runtime 1.24.3. For an app hosted below a URL prefix, pass absolute
URLs in `wasmPaths: {mjs, wasm}`. Vite apps can instead import the runtime files
with `?url` and pass those URLs; see `browser/src/sdk-demo.ts` in the project.
The SDK itself does not depend on Vite-specific imports.

Model URLs may be same-origin or CORS-enabled HTTP(S) directories. Use versioned,
immutable directories. The manifest is re-fetched and validated; cached artifacts
are namespaced by model URL and manifest contents. Model byte sizes are checked,
but this is not cryptographic authentication of downloaded weights. A successful
cached reload still contacts the host for the small manifest and runtime files.
Browser storage can be evicted or unavailable; loading continues without caching.

No prompt is sent to an AI server. The app's host serves the model and runtime
assets. Loading has no implicit Hugging Face download, telemetry or CDN fallback.
See the upstream [ONNX Runtime deployment guide](https://onnxruntime.ai/docs/tutorials/web/deploy.html)
for runtime asset serving and `wasmPaths`.

Importing the package is safe during server-side rendering, but **inference requires
a browser with WebGPU and shader-f16**. This package does not provide a Node GPU
backend or CPU fallback. Load it from client-side code, after hydration.

## Model limits and results

V3 supports 1–32 questions, 2–8 choices/ordered levels, and 1,024 tokens per complete
field prompt (shared context + question + choices + formatting). Oversized typed
requests fail rather than silently truncate. This is a local compatibility layer,
not Jev's private model or exact confidence formula. Fields are independent.

- `noul` is P(yes), not a boolean. A conventional boolean is `noul >= 0.5`.
- Choice returns an allowed label and its categorical distribution.
- Score is the expected index of the supplied ordered levels.
- Choice/Score confidence is one minus normalized entropy, not P(correct).

The V3 INT4 model achieved 83.0% on 1,600 held-out public cases and 59.6% field
accuracy on the known Jev-style workflow suite. It remains experimental. Before
the SDK extraction, a warm four-field browser request took about 92 ms in fast
mode and 148 ms in accurate mode on an M4 Mac mini; download and first shader
compilation are excluded. See the project results and SDK verification report.

## Development

```sh
cd sdk
npm ci --ignore-scripts
npm test
npm pack
```

The original browser playground consumes `decision-lab-sdk/engine`; the SDK loader
and the playground share the same inference implementation. The `/sdk.html` demo
exercises the public API, caching, cancellation, limits and disposal on real GPU
weights. No model weights are bundled in the npm package.
