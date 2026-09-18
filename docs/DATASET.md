# Decision Lab V3 dataset

205,058 labeled records for small Jev-inspired typed decision models. English only.
The records retain provenance and label support. Public annotations are not replaced with invented rationales.

| Split | Public | Generated | Total |
| --- | ---: | ---: | ---: |
| train | 159,097 | 39,961 | 199,058 |
| dev | 600 | 600 | 1,200 |
| calibration | 800 | 800 | 1,600 |
| test | 1,600 | 1,600 | 3,200 |

## Task families

| Task | Training records |
| --- | ---: |
| absa | 1,644 |
| ag_news | 36,000 |
| banking77 | 9,817 |
| boolq | 7,214 |
| emotion | 15,792 |
| piqa | 14,630 |
| snli | 50,000 |
| sst2 | 24,000 |
| synthetic_aspect | 5,000 |
| synthetic_email | 4,995 |
| synthetic_qualification | 4,999 |
| synthetic_risk | 4,992 |
| synthetic_routing | 4,992 |
| synthetic_support | 4,991 |
| synthetic_tool_policy | 4,992 |
| synthetic_verification | 5,000 |

## Record format

`id`, `split`, `task`, `source`, `synthetic`, `context`, `context_group`, `question`,
`options`, `target`, `answer`, `decision_type`, `supported_readouts`, and `evidence`.
Ordered decision exercises additionally include `score_target`.

For public data, `evidence` records the dataset, pinned revision, original row number,
official source split and original correct answer. These labels can contain annotation
noise. They are not claims that an independent reviewer verified every label.

For generated exercises, `evidence` contains the scenario facts, the governing rule
or question, and the derived answer. No paid teacher model was used. Generated labels
are programmatic; source code is included for inspection.

## Sources

- [boolq: google/boolq](https://huggingface.co/datasets/google/boolq), revision `35b264d03638db9f4ce671b711558bf7ff0f80d5`.
- [sst2: stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2), revision `8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb`.
- [ag_news: fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news), revision `eb185aade064a813bc0b7f42de02595523103ca4`.
- [snli: stanfordnlp/snli](https://huggingface.co/datasets/stanfordnlp/snli), revision `cdb5c3d5eed6ead6e5a341c8e56e669bb666725b`.
- [banking77: mteb/banking77](https://huggingface.co/datasets/mteb/banking77), revision `18072d2685ea682290f7b8924d94c62acc19c0b2`.
- [emotion: dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion), revision `cab853a1dbdf4c42c2b3ef2173804746df8825fe`, subset `split`.
- [piqa: nthngdy/piqa](https://huggingface.co/datasets/nthngdy/piqa), revision `467437b6dc793b01c07946dd3e800b9bd3199993`.
- [absa: tomaarsen/setfit-absa-semeval-restaurants](https://huggingface.co/datasets/tomaarsen/setfit-absa-semeval-restaurants), revision `8885372fe73256f96bb60f65b550538ac5c26047`.

## Quality controls and limits

- All answer indices are in range and match the stored answer.
- Zero exact duplicate prompts remain in any split; 39 were removed from training.
- Zero normalized-context overlaps between splits. Context hashes lowercase and normalize whitespace.
- Fresh calibration and test exclude all previously inspected V1/V2 contexts.
- Records sharing the same normalized context stay within one split. This does not prove the absence of semantic near-duplicates in source corpora.
- The ABSA source has blank test labels. Its evaluation examples are a context-group holdout from labeled training data.
- Banking77 is candidate selection with 4, 6, or 8 offered intents, including two lexical hard negatives. It is not full 77-way classification.
- Generated yes/no examples are approximately balanced within each synthetic task.
- A tokenizer-only length audit found two training records whose question and choices exceed the 1,024-token model budget. They remain in this general dataset, but are excluded from this model run after its step-2,800 recovery boundary. No development, calibration, or test question/options exceed the checked budget. Long context can still exceed the full request budget.
- Synthetic evaluation uses separate wrapper templates, entity names and sentiment vocabulary, but the same generator logic. Report it separately from public data and real-world evaluation.
- This dataset does not establish broad prompt-injection resistance, multilingual ability, legal expertise or production reliability.
- Upstream datasets retain their own licenses and terms. This compilation does not relicense them.
- `research/reports/v3/source-cards.json` records license and language metadata from the pinned upstream dataset cards; consult those sources before redistribution or commercial use.

## Training boundary

`training-data.json.gz` contains only training and development records, with compact columns.
Detailed JSONL files include calibration and test for reproducibility, but those splits
are not made available to the checkpoint-selection training process.
The bounded Colab pass samples from the large training corpus; it is not a claim that
every record is consumed in one free GPU session.

## Reproduce

Run `python -m decision_lab.data.build` from the repository root with the pinned Python dependencies
and included `research/evals/v3-exclusion-contexts.json` hash manifest. The builder refuses to overwrite a frozen run.
Included audit files document the actual produced dataset.
After building, run `python -m decision_lab.data.audit runs/v3/dataset` from the
repository root to check labels, provenance, duplicate prompts and split overlaps.
The audit itself uses only Python's standard library.

Full data SHA256: `be3b54f8361922d94344f8accf106b9b52fb925322dd80588e81b104cd70877c`
