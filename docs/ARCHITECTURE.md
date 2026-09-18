# How Decision Lab works

The application already knows the possible answers. The model's job is to
assign scores to them. This changes the output stage of an adapted language
model into a constrained decision interface.

## Example

For the context “The plot was awful, but the acting was excellent,” an app may ask:

- Was the acting good? → No / Yes.
- What is this about? → Movies / Sports / Politics.
- How good was the acting? → Bad / Mixed / Good.

Each question is converted into a multiple-choice prompt. Labels map to A, B,
C, and so on; the original label can be several words long. The model only
scores the corresponding answer rows. Application code maps the score back
to the original value.

## Backbone and trained head

The prototype uses LiquidAI/LFM2.5-350M, not Jev's private model. It is a hybrid
decoder with 16 layers: six grouped-query attention blocks and ten short
convolution blocks. Its hidden width is 1,024 and vocabulary size is 65,536.
See the [upstream model card](https://huggingface.co/LiquidAI/LFM2.5-350M).

Training adds rank-16 LoRA updates to linear layers and an eight-row output
projection initialized from the pretrained embeddings for the single tokens
A–H. Around six million parameters are trainable. The backbone is adapted
with supervised cross-entropy; the output head is not trained from scratch
against the full vocabulary.

For a final hidden vector `h`, the head computes:

```text
logits = W_choice · h              # W_choice has 8 rows
logits[unused_choices] = masked
probability = softmax(logits / T)  # T fitted on a separate calibration split
```

The runtime never needs to compute a full 65,536-token output distribution.
This does not remove the backbone cost: every suffix still runs through the
model. Training itself uses regular complete prompts; shared-prefix caching is
an inference optimization checked for numerical agreement.

## Shared prefill and independent suffixes

The runtime tokenizes all field prompts, including the optional reversed
choice orders, and finds their longest common prefix. It keeps at least one
token in each suffix. In ordinary use the common prefix contains the chat
instructions, context, and any shared question prefix.

It evaluates this prefix once. Each suffix sees the same prefix state, with
no answers from other fields. That is why fields can run in batches.

There are two kinds of state to preserve:

- Attention keys and values, which grow with the prefix sequence.
- Short-convolution state, which contains the relevant preceding activations.

Both must be forked or broadcast correctly. Reusing only a Transformer-style
KV cache would be wrong for this hybrid backbone. `portable_model.py` makes
the state tensors explicit for ONNX. The native cache helpers and portable
graph have small numerical tests, including branch isolation.

The default field batch is eight sequences. More questions or two option
orders can require multiple suffix batches. This is one shared prefill plus
one or more suffix calls, not one total forward pass.

## Typed answers

| Type | Readout |
| --- | --- |
| `choice` | Highest-probability allowed label plus the distribution |
| `noul` | Probability of Yes, between 0 and 1 |
| `score` | Expected ordinal index: sum of `index × probability` |

A Noul value of 0.02 means the model assigns about 2% probability to Yes. It
does not mean “2% confidence in the returned answer.” A conventional boolean
threshold is 0.5, but the appropriate decision rule depends on the application.

Score is derived from classification probabilities; it is not a separately
trained regression head. The selected ordinal scale matters. A score of 1.5
on three levels is not automatically an interval-scale measurement.

Choice/Score `confidence` is `1 - entropy(p) / log(number_of_choices)` in this
compatibility layer. This is not Jev's private confidence implementation and
is not a probability of correctness. High-confidence mistakes remain.

## Option order and calibration

The fast mode scores one order. The accurate mode also reverses the choices,
maps the resulting distribution back to the original order, and averages the
two probability distributions. This reduces one source of positional bias,
but adds work and does not guarantee higher accuracy on every example.

A global temperature is fitted using calibration examples after development
checkpoint selection. INT4 has its own temperature. A calibration result on
these tasks is not proof of calibration on unseen domains or every JSON schema.

## Browser deployment

LoRA updates are merged for export. The portable ONNX graph exposes its caches
as inputs and outputs, allowing one common-prefix call followed by batched
suffix calls. The chosen artifact quantizes matrix weights to four-bit blocks
of 32 while keeping embeddings, activations and caches in FP16 and the answer
head in FP32. Its weight file is about 296 MB.

The TypeScript SDK loads the manifest, tokenizer, graph, and external weights.
It checks declared sizes, optionally caches artifacts, creates a WebGPU session,
queues concurrent requests, and releases the session on disposal. It does not
generate text or contact an inference API. ONNX Runtime support files are still
required even when computation runs on WebGPU.

The original playground and SDK import the same inference implementation.
Browser fixture checks verify exact tokenization, class agreement, and the
full-context versus shared-cache path. Floating-point backends can differ near
decision boundaries; valid JSON is not a numerical or semantic accuracy proof.

## What this experiment does not establish

- That this implementation is better than Jev or reproduces its internals.
- That the method is novel, or that constrained decoding alone teaches reasoning.
- That a 350M model can perform arbitrary schema extraction or broad agent decisions reliably.
- That independent field scores satisfy cross-field constraints.
- That every request takes the same time or uses constant memory.
- That the upstream backbone's longer context is validated for this export.

The weakest observed result is the known workflow suite: 59.6% field accuracy
and 19.0% exact records. Improving that with fresh, reliable evaluation is the
main research opportunity.
