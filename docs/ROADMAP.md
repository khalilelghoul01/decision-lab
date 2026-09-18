# Research directions

These are opportunities, not promised features or measured improvements.

1. **Workflow accuracy first.** Collect a fresh, independently reviewed test
   beyond the 84 known scenarios. Track field accuracy and whole-record accuracy.
2. **Better data.** Hard negatives, negation, ambiguous instructions, realistic
   longer contexts, and evidence-supported labels. Test generator generalization.
3. **Confidence and abstention.** Domain-shift tests, selective prediction,
   calibration by task, and thresholds chosen without test-set tuning.
4. **Controlled ablations.** Head-only versus LoRA, data mixtures, model sizes,
   option ordering, and full-vocabulary versus small-head scoring.
5. **Cross-field consistency.** Compare independent scoring with explicit
   constraints, dependencies and staged decisions.
6. **Longer context.** Validate accuracy, memory and latency before raising
   the current 1,024-token input limit.
7. **Deployment.** More devices, browser memory profiling, cold-load benchmarks,
   worker execution, smaller assets and calibrated lower-precision exports.

The current small model's weak workflow performance is a useful negative
result. Faster valid JSON alone does not solve decision quality.
