# Decision Lab V3 results

V3 is a trained small decision model, with a working Python and browser runtime. It improves the measured public-task baseline and fixes a known acting regression. **It remains inadequate as a general Jev replacement:** the known workflow suite has only 57.5% native FP16 field accuracy and 59.6% for the selected compressed browser model. No claim of globally best accuracy or speed is supported by this experiment.

## Fresh final test

Checkpoint selection used development data only. After selection, a separate 1,600-example calibration split fitted a global temperature. The final test has 1,600 public examples and 1,600 generated rule exercises, 200 per task family.

| Model and inference | Public accuracy | Generated exercise accuracy |
| --- | ---: | ---: |
| Unmodified 350M base, one order | 55.44% | 47.44% |
| Unmodified 350M base, two orders | 60.88% | 55.19% |
| Trained V3 FP16, one order | 82.94% | 99.44% |
| Trained V3 FP16, two orders | **83.13%** | **99.44%** |
| Trained V3 INT4, one order | 82.31% | 99.13% |
| Trained V3 INT4, two orders | **83.00%** | **99.13%** |

The FP16 results use CUDA autocast on the Colab T4. The INT4 results use the portable ONNX artifact on CPU; all 3,200 final cases were evaluated after its selection. WebGPU was separately checked against that artifact and on the known workflow suite below, rather than rerunning the entire final test in the browser. There were no truncated final-test inputs. Generated tests share generator logic with training, so their high accuracy is evidence about these exercises, not broad real-world reliability. Public datasets may already have appeared in base-model pretraining; this project cannot verify upstream contamination.

| Public task, two orders | FP16 | INT4 |
| --- | ---: | ---: |
| Banking77 offered-candidate selection | 95.0% | 94.0% |
| AG News | 88.5% | 88.0% |
| Emotion | 88.0% | 88.5% |
| SST-2 | 86.5% | 85.0% |
| SNLI | 84.0% | 83.0% |
| Aspect sentiment | 79.0% | 79.0% |
| BoolQ | 77.0% | 76.0% |
| PIQA | 67.0% | 70.5% |

Banking77 offers 4, 6, or 8 candidates; this is not a 77-way benchmark score. The aspect-sentiment source's test labels are blank, so its test cases are a disclosed context-group holdout from labeled training data.

## Comparison with V2

On the identical 2,876 final-test examples that fit V2's four-choice limit, public accuracy rises from **75.47% to 81.19%**. This subset excludes Emotion and larger Banking77 candidate sets, so its public score differs from the full V3 test. Across public and generated cases combined, V3 gets 807 examples right that V2 misses, while V2 gets 88 right that V3 misses. V3's improvements combine a larger backbone, more varied training data, and training changes; this does not isolate the effect of model size.

The older V2 result of 83% was on a different four-task test and is not a valid direct comparison with V3's eight public tasks.

## Jev workflow regression and browser checks

The 84 workflow cases were previously inspected. They are a regression suite, not a new blind test and not a comparison with the real Jev API.

- V3 CUDA, questions-only, two orders: **138/240 fields (57.5%)**, **11/84 exact records (13.1%)**.
- V2's previously recorded comparable questions-only path: 52.5% fields, 15.5% exact records. Field accuracy improves, but exact-record accuracy does not.
- Actual V3 FP16 WebGPU: **136/240 fields (56.7%)**, 10/84 exact records, 84/84 valid accepted JSON outputs, no core input truncation.
- Browser versus native FP16: 238/240 core decisions agree. Two decisions differ near classification boundaries; probability differences reach 0.073. Do not assume identical probabilities across FP16 backends.
- The six oversized browser stress requests are explicitly rejected rather than silently dropping evidence. Native batch evaluation of those stress cases truncates context and is reported separately.

The regression text, `wasn't good enough, the acting bad so bad`, now produces P(good acting) = **0.00281** in the browser. For `The plot was awful, but the acting was excellent.`, P(good acting) = **0.98344**; an eight-choice topic question correctly selects Movies. These are diagnostic examples, not additional blind accuracy evidence.

## Calibration

FP16 temperature: **1.3634747716**. Calibration NLL falls from 0.24939 to 0.23580. On the fresh public test, NLL is 0.41831, multiclass Brier score is 0.23834, and ten-bin ECE is 0.02431. There are still **28 incorrect public predictions with selected-class probability at least 0.9**.

For the separately calibrated INT4 artifact, fresh public-test NLL is 0.43550, Brier score is 0.24910 and ten-bin ECE is 0.02003. It still has **35 public errors with probability at least 0.9**. Similar average accuracy does not establish equally reliable individual probabilities.

Noul is P(yes). Choice/Score `confidence` in this local compatibility layer is one minus normalized entropy, not a probability that the answer is correct. TypeSafe's exact confidence implementation is not reproduced.

## Measured speed

Apple M4 Mac mini, 16 GB. Warm inference; model download and initial shader compilation are excluded. Native timings include prompt encoding; browser runtime checks time prepared model inputs and exclude tokenization. These are different measurements and should not be directly equated.

| Native FP16, four fields | V2 shared cache | V3 shared cache |
| --- | ---: | ---: |
| Short context, one order | 44.2 ms | 62.3 ms |
| Short context, two orders | 67.5 ms | 97.6 ms |
| Longer context, one order | 100.9 ms | 139.8 ms |
| Longer context, two orders | 132.3 ms | 176.3 ms |

The longer native workload has a maximum 403-token field prompt in both models. For V3 with eight fields and two orders, shared caching reduces native time from **1,560 ms to 277 ms (5.64x)**.

Actual browser FP16, four fields and two orders: **166 ms** for the short fixture and **264 ms** for the longer fixture. The longer fixture falls from 895 ms without shared prefill to 264 ms with it (3.39x). Python/JavaScript tokenization matches exactly; the fixture's largest probability difference is 0.00082 and full/cache browser probabilities agree.

## Smaller browser artifact

The selected browser profile is `v3-q4`: **295,991,296 bytes** of weights, compared with **708,999,168 bytes** for FP16, a **58.3% reduction**. Constant matrix multiplications use symmetric four-bit blocks of 32; embeddings, activations and caches retain FP16, and the small answer head retains FP32.

Selection was recorded before its final test. Public calibration accuracy changes from 83.75% to 83.00%, within the predeclared one-percentage-point allowance. Generated calibration accuracy changes from 99.00% to 99.125%. The separate temperature is **1.2975177151**. Actual WebGPU fixture checks pass: exact tokenization, matching class answers, maximum probability difference 0.00056 against the corresponding quantized CPU reference, and identical full/cache browser probabilities.

For the same prepared four-field browser fixtures, INT4 takes **160 ms short / 262 ms longer**, versus FP16's 166 / 264 ms. These small timing differences do not establish a meaningful quantization speed gain. The main benefit is smaller assets. Shared prefill still reduces the longer INT4 fixture from 929 ms to 262 ms.

For the complete original four-field API request, including JSON parsing, validation, tokenization and response assembly, five warm runs have medians of **92.4 ms with one order** and **148.4 ms with two orders**. The background CPU test was paused for those timings.

On the known workflow suite, actual INT4 WebGPU scores **143/240 fields (59.6%)** and **16/84 exact records (19.0%)**, with valid JSON for all 84 accepted core cases. It gets 9/18 injection-stress fields right and rejects all six oversized requests. These numbers do not establish prompt-injection resistance. Quantized probabilities can differ substantially from FP16: the largest observed core difference is 0.402. They are not interchangeable confidence estimates.

The original negative-acting case has P(good acting) = 0.01668 in two-order INT4 mode and 0.02827 in one-order mode. The positive-acting/negative-plot contrast gives 0.96694 in two-order mode. The retained flat JSON Schema interface now uses V3's same question-only prompting and returns `good_acting: false` on the original example.

## Training and artifacts

- Base: [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M), pinned revision `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`.
- Hybrid decoder: 16 layers, six attention and ten short-convolution layers. About 354.49M merged parameters; 6.00M trainable parameters during adaptation.
- Rank-16 LoRA and eight trainable answer-token rows initialized from A–H. Cross-entropy supervised training followed by calibration; no RLCD claim.
- 3,200 updates, effective batch 32. Selected checkpoint: step 2,800. It used 89,600 draws covering 61,521 unique training records. The complete run sampled 102,400 draws and 67,393 unique records from the 199,058-record training set.
- The recorded training clock is 50.7 minutes, excluding setup and interrupted/discarded work. Recovery handled an initial gradient overflow and two oversized training questions. The CLI proxy connection also required refresh; the GPU job survived that connection issue.
- Selected weights and evaluation logs are downloaded and hash-verified. The allocated Colab runtime has been stopped.

The dataset has **205,058 detailed labeled field decisions**, with provenance, source revisions, generated-rule support, disjoint normalized-context splits, and audit scripts. Exact-overlap checks do not establish semantic deduplication. Dataset sources retain their individual licenses. Model artifacts retain the upstream LFM Open License and modification notices.

The runtime scores field suffixes after one shared context prefill. It maps choices to A–H, so multi-token label names do not require autoregressive label generation. Both attention and convolution caches are handled. This is one prefill plus suffix processing, not literally one total forward pass. Fields are independent, so schema validity does not ensure cross-field consistency.

Local limits: 2–8 choices or Score levels, 32 questions, 1,024 tokens per full field prompt. The typed API rejects oversized requests. It supports Jev-shaped Choice, Noul and Score responses, not arbitrary free-text extraction or Jev's private model, service, billing, or exact confidence formula.

Aggregate evidence is in [reports/v3](../reports/v3/), including `comparison.json`,
`q4-test-accurate.json`, `jev-regression.json`, `browser-q4-jev.json`, and the
native/browser benchmark reports. Per-example predictions are a separate
`decision-lab-evaluation-records.zip` asset in [Releases](https://github.com/khalilelghoul01/decision-lab/releases).
INT4 final-test public accuracy is 0.125 percentage points below FP16. The
smaller artifact passed the recorded calibration and browser checks and is the
selected research release; this does not remove the workflow limitations above.
