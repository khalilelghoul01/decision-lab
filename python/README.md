# Python runtime and research tools

Run commands from the **repository root**. Use Python 3.12 or 3.13.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e './python[dev]'

python scripts/download_model.py --native
python -m decision_lab.protocol examples/jev-review-request.json --checkpoint runs/v3
```

The native checkpoint contains adapters and an answer head. On first use,
the runtime also downloads the pinned Liquid AI base model.

## Find the code

| Module | Purpose |
| --- | --- |
| `decision_lab/engine.py` | Shared-prefix inference and JSON Schema decisions |
| `decision_lab/protocol.py` | Choice, Noul and Score request/response handling |
| `decision_lab/portable.py` | Exportable attention and convolution caches |
| `decision_lab/train.py` | Current V3 model and supervised training |
| `decision_lab/evaluate.py` | Calibration, baseline and final evaluation |
| `decision_lab/data/` | Dataset builders, synthetic rules and audits |
| `decision_lab/export/` | ONNX export, quantization and parity checks |
| `decision_lab/schemas/` | JSON contracts shared with the TypeScript SDK |
| `decision_lab/legacy/` | V1/V2 utilities needed for historical baselines |
| `tests/` | Native runtime, training-math and download tests |

Earlier experiments are retained for reproducibility. Start with `train.py`
for the released model. [Full training guide](../docs/TRAINING.md).

## Tests

```sh
pytest -q -c python/pyproject.toml
```

These tests use small fixtures and do not download model weights. One test
requires historical V1/V2 data and skips when those files are absent.
