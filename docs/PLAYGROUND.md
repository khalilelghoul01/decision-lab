# Browser playground

The public playground is available at
[decision-lab.loomens.com](https://decision-lab.loomens.com). It is a static
Cloudflare Workers application. Model inference happens in the visitor's WebGPU
browser; Cloudflare only serves the application and model artifacts.

## What is included

- Builder, Jev-shaped request JSON, and flat JSON Schema inputs.
- Choice, Noul, and Score answers with complete probability distributions.
- Fast one-order and more stable two-order scoring.
- In-memory run history and JSON exports.
- Warm on-device latency benchmark with individual samples.
- Exact tokenizer, Python-reference, and shared-cache parity checks.
- The frozen 84-case workflow regression suite plus 12 stress cases.
- Architecture notes, SDK example, licenses, and links to the training evidence.

The evaluation suite is labeled as a known regression suite. It is not a fresh
blind evaluation and it does not call Jev. Confidence is also labeled as a model
readout rather than a probability that the decision is correct.

## Cloudflare layout

The Worker serves the compiled Vite application through Static Assets. Files
larger than the static-asset limit are stored in the dedicated
`decision-lab-models` R2 bucket:

```text
decision-lab.loomens.com
├── Worker Static Assets: HTML, CSS, JavaScript, evaluation fixture
└── R2 binding: model graph, 296 MB weights, tokenizer, ONNX WASM runtime
```

R2 responses support `GET`, `HEAD`, and byte-range requests, carry immutable
cache headers for versioned binaries, and are returned from the same origin.
The application sends COOP and COEP headers required by the runtime.

## Reproduce the deployment

Authenticate Wrangler, download the public model into
`browser/public/model-v3-q4`, and run:

```sh
cd browser
npm ci
npm run build
npx wrangler r2 bucket create decision-lab-models  # first deployment only
npm run cf:upload
npm run cf:deploy
```

`cf:upload` preserves the public model directory structure and uploads the
hashed ONNX runtime binary under its generated URL. The build's `.assetsignore`
keeps those large objects out of the Static Assets upload.

The custom-domain route is declared in `browser/wrangler.jsonc`. Forks should
change the Worker name, R2 bucket, and route before deploying.

## Browser requirements

The released runtime needs HTTPS or localhost, WebGPU, and `shader-f16`. The
initial transfer is roughly 301 MB and can be cached by the SDK. The first
inference may compile GPU shaders; use the test bench for warm timings on the
actual device.
