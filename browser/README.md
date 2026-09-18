# Browser demo

Requires Node.js 20+ and a browser with WebGPU and `shader-f16`.
Run from the repository root:

```sh
python3 scripts/download_model.py
npm --prefix sdk ci --ignore-scripts
npm --prefix sdk run build
npm --prefix browser ci --ignore-scripts
npm --prefix browser run dev
```

Open the printed localhost URL. The main page is the decision playground;
`/sdk.html` demonstrates loading, progress, cancellation and GPU cleanup.
Prompts stay in the browser. Initial loading includes about 296 MB of weights
and shader compilation.

```sh
npm --prefix browser run build
```

The static production build is written to `browser/dist/`. Serve it over
localhost or HTTPS. [Use the SDK in another app](../sdk/README.md).
