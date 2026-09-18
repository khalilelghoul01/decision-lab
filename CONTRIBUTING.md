# Contributing

This is a small, early research experiment. Focused improvements and honest
negative results are welcome. Please open an issue for a substantial change
before spending time on a large implementation.

## Good first contributions

- Add independently labeled decision cases that expose a real failure.
- Explain a wrong result with a minimal request, expected answer and rationale.
- Test the model loader on another browser/device and report the environment.
- Improve error handling, memory use, docs or TypeScript ergonomics.

## Development checks

```sh
cd sdk
npm ci --ignore-scripts
npm test
cd ../browser
npm ci --ignore-scripts
npm run build
```

Python setup is in [Training](docs/TRAINING.md). Run `pytest -q` after changing
the native runtime, training math or protocol. Small tests should not download
full model weights. Changes to the graph, tokenizer, prompt format or caches
also need real native/ONNX/browser parity checks.

TypeScript and browser source use Prettier 3.6.2. Generated schemas and vendored
notices are excluded in `.prettierignore`. Keep formatting changes separate from
behavior changes when possible.

## Research changes

Keep train, development, calibration and final test separate. Once a test has
been inspected to choose a change, treat it as a known regression set and
create a fresh test for the new claim. Report public and generated cases
separately. Include baseline, model version, device, precision, context length,
field count, option-order mode, and whether loading/compilation is timed.

If a change improves one metric and worsens another, report both. Do not claim
to outperform Jev without a reproducible comparison against its actual service.
Do not silently replace the historical reports with new results.

## Pull requests

Describe the problem, the resulting behavior, and the checks you ran. Include
a minimal example for API changes. Keep unrelated formatting out of the diff.
Never commit credentials, private customer examples, downloaded datasets,
checkpoints, caches, or built browser bundles. Add source attribution and
license information for any new data or code.

Contributions to original project source are under the repository's MIT
license. Model weights and public data retain their separate upstream terms.
