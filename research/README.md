# Research evidence

Start with the [results summary](../docs/RESULTS.md) or
[dataset card](../docs/DATASET.md). This directory preserves the underlying
records; it is not needed to use the browser SDK.

| Directory | Contents |
| --- | --- |
| [reports/v3](reports/v3/) | Frozen metrics, training configuration, calibration, benchmarks and data audits |
| [reports](reports/) | SDK browser checks and original release verification |
| [evals](evals/) | Known workflow regression cases and earlier-context exclusion hashes |

The reports describe the original experiment. Paths recorded inside historical
JSON files reflect that run's layout; they are evidence, not current commands.
The current commands are in the [training guide](../docs/TRAINING.md).

The workflow suite was previously inspected, so it is a regression set rather
than a blind benchmark. Generated test exercises share generator logic with
training. Public-task and generated results must be reported separately.

Per-example predictions are available in the evaluation-records archive on the
[release page](https://github.com/khalilelghoul01/decision-lab/releases/tag/v0.1.0).
Full training data and model weights are excluded from Git.
