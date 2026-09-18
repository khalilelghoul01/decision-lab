# Reproduce or extend the experiment

The pipeline targets Python 3.12/3.13, PyTorch and a CUDA GPU for training.
The recorded run used a free Colab T4 with roughly 15 GB usable VRAM. A free
runtime is not guaranteed to remain allocated. This repository does not need
SSH or any private API credentials.

The recorded run is evidence of one execution, not a promise of bit-for-bit
determinism across GPU kernels, package versions, or machines. The source
revisions, data audits and selected configuration are in `reports/v3`.

## Environment

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0
pip install -r requirements-engine.txt
pip install pytest
```

On Colab, use its supplied PyTorch build if compatible. The original T4 run
used torch 2.11.0+cu128, transformers 5.17.0 and PEFT 0.21.0. Dataset construction
used the pinned dependencies in `requirements.txt`; Colab's already-built
training payload did not need to regenerate the dataset there.

## Build and audit the data

```sh
python build_v3_data.py
python audit_v3_dataset.py runs/v3/dataset
python check_v3_lengths.py --root runs/v3
```

The builder retrieves public datasets at pinned revisions and generates rule
exercises. It refuses to overwrite an existing frozen `runs/v3/data.json`.
The included exclusion file contains normalized-context hashes from earlier
experiments, not those experiments' full data.

Generated outputs include detailed JSONL splits, `data.json`, and
`training-data.json`. The training payload contains only train and development
records. Calibration and final-test labels are not passed to the trainer.

The recorded compilation had 205,058 records. The dataset card explains source
terms, the ABSA holdout exception, Banking77 candidate selection, duplicate
handling and the limits of synthetic evaluation. Consult source licenses
before redistributing generated compilations of the public data.

The length audit writes `training-eligibility.json`. For a new run it excludes
oversized question/choice prompts from step zero. The historical run discovered
two such examples later and applied the exclusion after step 2,800. That
difference is intentional and documented, so a new run is not an exact replay
of the interrupted training history.

## Train on the GPU

```sh
python decision_v3.py \
  --data runs/v3/training-data.json \
  --output runs/v3 \
  --steps 3200 \
  --batch-size 8 \
  --accumulate 4 \
  --eval-every 200 \
  --max-minutes 150
```

Sampling balances the 16 task families and permutes option order. Effective
batch size is 32. The trainer saves recovery state and selects its best
development macro accuracy. To resume the same configuration, repeat the
command with `--resume`; do not overwrite the frozen data or change the options.

The historical selected checkpoint was step 2,800, covering 61,521 unique
training records. The 3,200-step pass sampled 67,393 unique records in total.
The dataset size must not be reported as the number of unique examples
actually consumed by the selected checkpoint.

## Calibrate and evaluate after selection

```sh
python scripts/prepare_evaluation.py --root runs/v3
python evaluate_v3.py \
  --root runs/v3 \
  --data runs/v3/evaluation-data.json \
  --suite evals/jev-usecases-v2.json
```

Evaluation requires `training-complete.json` and refuses to overwrite its final
summary. It fits temperature on calibration data, evaluates the selected model
and unadapted baseline, and reports public and synthetic tasks separately.
The known workflow regression is not a blind test.

The old V2 common-subset comparison needs that separate historical checkpoint
and its files. It is not required to train or evaluate V3.

## Export, compress and verify

```sh
python export_browser.py --checkpoint runs/v3 --output browser/public/model-v3
python quantize_browser.py browser/public/model-v3 browser/public/model-v3-q4
python validate_quantized.py --root runs/v3 --artifact browser/public/model-v3-q4
```

Use calibration results and browser reference checks to choose an artifact
before inspecting its final test. The historical gate allowed at most one
percentage point of calibration accuracy loss in both public and synthetic
task averages. Record your choice before running:

```sh
python validate_quantized.py --root runs/v3 --artifact browser/public/model-v3-q4 --final-test
python export_jev_browser_suite.py --checkpoint runs/v3
```

Run the playground's **Run checks** and **Test Jev use cases** buttons on the
target browser/device. The SDK example has a separate lifecycle check. Report
first-load time, warm latency, model size, accuracy and calibration separately.
Do not advertise a speed improvement from file-size reduction alone.

## Use the released checkpoint in Python

```sh
python scripts/download_model.py --native
python jev_protocol.py examples/jev-review-request.json --checkpoint runs/v3
```

The checkpoint download contains adapters/head, configuration and calibration.
Python also retrieves the pinned Liquid AI base weights if they are not cached.
It is not an optimizer-state archive for resuming the historical training run.

## Tests without downloading model weights

```sh
pytest -q
cd sdk
npm ci --ignore-scripts
npm test
```

The Python tests use small synthetic model configurations and probability
fixtures. One historical-data test is skipped when the V1/V2 data are absent.
The JavaScript tests mock the runtime for lifecycle/error cases; the recorded
real-WebGPU checks are separate evidence in `reports`.
