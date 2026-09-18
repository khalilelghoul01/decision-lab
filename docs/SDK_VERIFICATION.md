# SDK verification

Decision Lab SDK 0.1.0 was checked on the M4 Mac mini with 16 GB of memory.

## Package and lifecycle

- 11 automated tests pass: URL and manifest validation, HTTP failures, byte-size
  checks, download progress, cache reuse/refresh/clearing, cancellation, request
  queuing, failed-request recovery, disposal and safe import without WebGPU.
- The generated npm tarball was installed in a separate consumer project.
  TypeScript accepted the public declarations under NodeNext resolution and
  rejected wrong result labels and invalid answer properties.
- Importing the installed package under Node works; loading there reports the
  expected `UNSUPPORTED_ENVIRONMENT` error. Browser GPU inference is required.
- The installed `decision-lab-assets` command copies both support files from
  the pinned ONNX Runtime package.
- The production browser build passes. The SDK owns the inference engine used
  by both the original playground and the new loader example.

## Real WebGPU checks

The `/sdk.html` example uses the public `loadModel()` and `systemOne()` APIs with
the actual V3 INT4 weights, rather than a mocked runtime.

| Check | Result |
| --- | --- |
| Original negative-acting example | P(yes) = 0.0166823 |
| Positive acting despite negative plot | P(yes) = 0.9669392 |
| Eight-choice topic | Movies |
| Score response | Numeric expected ordinal index |
| Oversized context | Rejected explicitly |
| Concurrent calls / queued cancellation | Pass |
| Dispose while requests are queued | Accepted work drains; session releases |
| Calls after disposal | Rejected |
| Reload from persistent browser cache | Tokenizer, graph and weights reused |
| Predictions after reload | Identical |
| Abort while reading model weights | Pass |

Five warm four-field requests in accurate mode measured 148.48, 144.57, 144.84,
142.99 and 143.80 ms, for a **144.57 ms median**. These include the public SDK call
and prompt encoding; model download, initial load and shader compilation are
excluded. This small sample is not a latency guarantee.

The [raw evidence](../research/reports/sdk-browser-checks.json) verifies runtime
behavior and known regressions; it is not a fresh model-accuracy benchmark.
Existing model accuracy and limitations remain in [Results](RESULTS.md).
