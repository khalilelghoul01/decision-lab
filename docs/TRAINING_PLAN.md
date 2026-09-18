# Decision Lab V3: frozen training and evaluation plan

Target: a small, local Jev-inspired typed decision model, trained on a free Colab T4 through the official CLI. No paid API teacher, no paid GPU, no
claim to reproduce Jev's unpublished training procedure.

## Model and inference

- LFM2.5-350M, revision `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`.
- Rank-16 LoRA on the linear layers, plus eight trained answer-token rows
  initialized from the pretrained A–H embeddings. No autoregressive output.
- 2–8 allowed choices, 1,024-token full field prompt budget. Shared context
  prefill and independent field branches at deployment. Jev-shaped Choice,
  Noul and Score outputs are assembled and validated in application code.
- Compare one option order with the original/reversed two-order average.

## Data

The builder freezes context-group-separated training, development, calibration,
and final-test files before training. Previously inspected V1/V2 examples are
excluded from fresh calibration/test. The old Jev suite is a known regression
set and must not be advertised as a new blind test.

The detailed JSONL dataset contains source/revision, source-row reference,
context, question, allowed choices, label, supported typed readouts, and either
the original annotation reference or the facts/rule that generated its label.
Public benchmark annotations and generated exercise labels are kept distinct.
Synthetic test exercises use separate wrapper templates and vocabulary but
share generator logic: their scores demonstrate performance within that
exercise distribution, not independent production accuracy.

## Training and selection

- Balanced sampling over 16 task families, random option permutations,
  approximate length bucketing, FP16 autocast with FP32 parameters.
- Cross-entropy supervised training. RL is not required for this known-label
  objective and is not being claimed; the earlier RL phase did not improve V2.
- Effective batch 32, up to 3,200 optimizer steps, 150-minute training cap.
  The cap excludes setup and final evaluation. This is a bounded training pass
  over the larger dataset, not a promise that every example will be sampled.
- Development evaluation every 200 steps; choose the best macro accuracy over
  the 16 families using one option order. Preserve the pretrained checkpoint
  if training never improves development accuracy. Save recovery state every
  100 steps. No final-test labels are uploaded to the training process.
- Fit one global scalar temperature on the separate calibration split only.
  Report accuracy, per-task results, NLL, Brier score, ECE, and high-confidence
  errors separately for public data and generated exercises.

## Final validation and handoff

After selecting weights, evaluate the untouched final test once per prespecified
inference mode. Compare with the unmodified 350M base and the earlier V2 on its
supported common subset. Report the known Jev regressions separately. Export
the selected candidate, check native/ONNX/browser parity, and benchmark warmed
inference. Keep V2 available until V3 is validated. Retrieve the checkpoint and
logs from Colab and stop the allocated runtime after completion.

## Optional smaller browser artifact (declared before final-test evaluation)

Try weight-only INT4 with 32-element blocks, preserving embeddings, activation
and cache precision, and the small FP32 answer head. Choose it only if separate
calibration evaluation loses no more than one percentage point on either the
public or synthetic task average, and real WebGPU parity checks pass. Measure
its browser latency; do not assume that a smaller file is faster to execute.
If it passes, report its final-test results separately from FP16. If it fails,
retain FP16. Do not tune quantization to final-test labels.
