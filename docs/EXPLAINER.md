# A short explanation you can share

I built Decision Lab as a small open-source experiment: can a language model
make structured decisions quickly without writing JSON one token at a time?

You give it context and questions with a fixed set of possible answers. It
reads the shared context once, reuses the cached state for each question,
scores the allowed answers, and lets normal code assemble valid JSON.

The prototype uses an adapted 350M-parameter model and runs in the browser
with WebGPU. On my M4 Mac mini, a warm request with four fields took roughly
92–148 ms, depending on the scoring mode. The compressed weights are about
296 MB, so the first download and load are a separate cost.

The interesting part is the tradeoff. It scored 83% on a held-out public-task
test, but only 59.6% of fields and 19% of whole responses on the known workflow
suite. Fast, valid JSON does not mean reliable decisions.

This is a concept to inspect and improve, not a production-ready replacement
for Jev. I published the code, model artifacts, training pipeline, SDK, tests,
and the disappointing results too. Contributions around better evaluations,
data, calibration and runtime performance are welcome.

[Repository](https://github.com/khalilelghoul01/decision-lab) ·
[Detailed architecture](ARCHITECTURE.md) · [Measured results](RESULTS.md)
