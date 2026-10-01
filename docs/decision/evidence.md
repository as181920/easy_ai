# Evidence-supervised decision experiment

This round compares answer cross-entropy (CE) with **answer CE + 0.2 × supporting-sentence CE**. It tests whether explicitly learning who a fact belongs to helps the shared encoder. Lateness examples are informal probes and are not acceptance criteria. No external weights, teacher responses, RL or keyword inference rules are used.

The parent models are our three scratch-trained public-supervision checkpoints at `runs/decision/semantic-coverage-v2/baseline-{1337,2027,3407}/selected-4000`. Both arms copy exactly the same parent tensors for each seed, initialize the same auxiliary head, reuse the public 12,000-token native byte-level BPE, and start fresh optimizer state. Main training starts from the parent again, not from the capacity-check checkpoint.

```text
ChoiceModel                                      6,693,377 parameters
|-- Existing shared encoder + matching scorer    6,627,841
|   |-- Token embedding: 12000 x 256
|   |-- Bidirectional Transformer blocks x 4
|   |   `-- 4 attention heads; FFN 256 -> 768 -> 256
|   `-- Candidate/state interaction block x 1
`-- Auxiliary evidence projection: 256 x 256         65,536

state ----------------------> shared encoder ------> token memory
question + each candidate --> shared encoder ------> query vectors
                                      |                    |
                                      +--> matching ------> answer scores --> CE
                                      |                    |
newline-derived sentence masks --> sentence means          |
                                      |                    |
                                      +<-- projected mean query
                                      |
                               sentence scores --> evidence CE

training loss = answer CE + lambda * evidence CE
answer arm: lambda = 0; evidence arm: lambda = 0.2
inference: answer scores -> softmax(temperature) -> Ruby Hash/JSON
```

Supporting-sentence indices are targets only. They never choose a sentence for the answer scorer, alter attention masks, or enter the forward query. Removing or changing an annotation leaves inputs and predictions unchanged. Sentence masks come from newline boundaries; both tokenizers expose byte offsets for alignment. These are structural boundaries, not semantic keyword rules. Inference still reads the full state and needs no annotation; the evidence head is not run by `Predictor`.

## Data and evaluation

`EasyAI::Decision::Data::EvidenceCorpus` generates explicit Boolean worlds first, then renders text and derives the answer and supporting sentence from those records. No model produces the gold labels. The templates are inspectable, but this is **controlled synthetic data, not a human-annotated natural-language benchmark**.

| Split | Rows | Independent actor-pair/action families |
|---|---:|---:|
| Controlled train | 24,576 | 64 |
| Controlled validation | 3,072 | 16 |
| Controlled calibration | 3,072 | 16 |
| Controlled novel-expression test | 6,144 | 32 |
| Familiar-expression test | 6,144 | Same 32 test families |
| Capacity check | 128 | Two training families |
| Fresh public test | 809 | 450 public groups |

Each family covers two subjects, affirmative/negated claims, all four truth assignments, Chinese/English, two sentence orders, and absent/positive/negative unrelated facts. Train uses two wording styles; validation/calibration use a third; novel test uses a fourth and alternative state verbs. Familiar test isolates expression shift on the same held-out families. These two tests are correlated and must not be counted as independent evidence. All translations and truth-flip variants of a family stay in one split. Actors and actions can be familiar; their held-out combinations are new.

Main training combines 107,094 existing public supervised examples with the 24,576 controlled examples: **131,670 rows / 38,477 groups**. Source-balanced sampling gives the controlled source roughly one quarter of training draws; it does not create new independent information. The same sample schedule is used in both arms. Unannotated public examples contribute answer CE; their evidence targets are ignored.

The fresh public test excludes all original public train/validation/calibration/test groups **and the preceding coverage experiment's challenge groups**. It contains 150 reserved groups per public source. Preparation checks token lengths and rejects silent truncation. Data and parent weights are fingerprinted in `protocol.json`.

Metrics include answer accuracy by language and condition, NLL/Brier/ECE, evidence accuracy, and all-correct groups for fact flips, question flips, subject switches, sentence order, irrelevant facts, and four-case binding. A four-case binding group contains two positive and two negative answers; predicting a majority label cannot pass it. The row count is much larger than the informal probes, but only 32 independent generated test families support the generalization claim.

## Fixed protocol

- Seeds: 1337, 2027, 3407; two arms; all six main runs retained.
- Capacity diagnostic: 300 updates on 128 training rows, seed 1337, both arms. Evaluate final weights on those same rows; this is fitting, not validation or generalization.
- Main budget: 800 updates per run, effective batch 32 (microbatch 16 × accumulation 2), learning rate 0.0001, warmup 50, FP32, original dropout 0.1.
- Select the main checkpoint by **answer-only validation NLL** on 384 source-balanced held-out rows (96/source). Auxiliary loss cannot choose a checkpoint.
- A single temperature is fitted on 384 source-balanced calibration rows (96/source). Test data never fits the temperature or selects weights/seeds.
- The evaluation entry point refuses to open tests until all six fixed training budgets finish. Jobs run in separate sequential processes to release CUDA allocations and avoid the previous concurrency OOM.
- Auxiliary-benefit gate: mean four-case binding gain at least 5 percentage points, improvement in at least two seeds, no mean per-language loss above 3 points against answer-only, no public-source macro loss above 3 points against parents, and no worse mean calibrated controlled/public NLL against answer-only.
- Separate readiness gate: every evidence seed reaches 95% per-language accuracy and 90% four-case binding correctness. Passing a relative-improvement gate alone does not make the model ready.

## Reproduce and inspect

Existing parent checkpoints and the local public corpus are prerequisites. From the repository root, one command prepares fresh artifacts, runs the capacity checks and six main fits sequentially, evaluates all three parents and all six trained models, and writes the comparison plot:

```bash
bundle exec ruby benchmarks/decision/evidence.rb --phase all --output runs/decision/evidence-v1-reproduction
```

Individual phases are `prepare`, `sanity`, `train`, `evaluate`, and `report`; training accepts `--arm answer|evidence --seed 1337|2027|3407`. Preserve interrupted directories rather than overwrite them. A failed run is evidence, not an excuse to silently choose a different budget.

```bash
tail -f runs/decision/evidence-v1/evidence-1337/train.log
bundle exec ruby benchmarks/decision/evidence.rb --phase report --output runs/decision/evidence-v1
```

Artifacts live under the ignored run directory: `protocol.json`, generated data and checksums, per-run logs/traces/checkpoints/reports, `evaluation-*.json`, `sanity-evaluation.json`, `report.json`, `comparison.tsv`, and `comparison.png`. Each training report retains the standard loss/validation and memory plots; the historical `learning/` gradient-descent illustrations remain unchanged.

Calibrated checkpoints live at `runs/decision/evidence-v1/{answer|evidence}-{seed}/calibrated`. They are experimental artifacts. Use the recorded results below to assess whether either arm improves behavior; do not pick a seed by its test score and present that as an unbiased result.

## Engineering lesson: optional parameters and resume

Public-only batches leave the auxiliary head unused. Retaining zero gradient tensors would advance its AdamW moments and weight decay after earlier annotated batches, while a resumed model starts with absent gradients and skips those updates. This broke exact resume. `AdamW#zero_grad` now clears gradients to absent values, preserving the same update semantics across interruption.

Torch.rb 0.23's `Parameter#grad=` native setter crashes on `nil`; its `Tensor` setter supports clearing. The optimizer explicitly binds the Tensor setter and clears only existing gradients. Regression tests cover unused parameters, mixed annotated/public batches, exact resume with dropout, annotation independence, offset alignment, blank sentence lines, corpus isolation, and the evaluation guard.

## Results

Completed; no meaningful benefit from evidence supervision at this recipe/budget. All six runs finished 800 updates on CUDA; answer-only validation NLL selected step 100 in every run. Do not interpret these as 800-update selected weights.

| Three-seed mean | Parent | Answer CE | Answer + evidence CE |
|---|---:|---:|---:|
| Novel-expression accuracy | 49.48% | 50.71% | 50.76% |
| Four-case binding all correct | 5.47% | 7.60% | 7.60% |
| Familiar-expression accuracy | 49.03% | 55.36% | 54.65% |
| Fresh public-source macro accuracy | 57.89% | 58.45% | 58.34% |

The original auxiliary-benefit and task-readiness gates both fail. Learning evidence selection did not establish answer reasoning. Early checkpoint selection and limited fitting mean this is a bounded negative result, not proof that evidence supervision can never help.

The supplemental natural-data panel also shows weak transfer:

| Task/language | Parent | Answer CE | Evidence CE | Majority | Chance |
|---|---:|---:|---:|---:|---:|
| AG News / English | 27.07% | 26.03% | 26.40% | 26.70% | 25.00% |
| Emotion / English | 16.00% | 14.03% | 13.97% | 36.20% | 16.67% |
| MASSIVE scenarios / English | 7.33% | 9.26% | 9.15% | 14.11% | 5.56% |
| MASSIVE scenarios / Chinese | 3.84% | 4.37% | 4.22% | 14.22% | 5.56% |

Reducing the 95% diagnostic threshold does not resolve these gaps. Broad data/task coverage and independent generalization evaluation should guide the next training plan. Validation-boundary process memory observations ranged from 1472 to 1700 MiB; all main runs stayed on CUDA. These observations are not a measured peak-memory bound.

Artifacts: `runs/decision/evidence-v1/report.json`, `comparison.png`, `loss-components.png`, `generalization/report.json`, and `generalization/summary.txt`. The loss-component figure separates answer CE from auxiliary CE; training curves use 25-update moving means and validation points are unsmoothed.

## Interpreting thresholds against published models

Checked 2026-09-30. Our 95% language threshold is a chosen reliability target for simple, fully specified explicit facts. It is not a standard Jev benchmark, a universal threshold for intelligence, or sufficient evidence of general semantic understanding. Failing it demonstrates a weakness on this controlled task; passing it would establish only this task's reliability.

| Published benchmark | Model | Reported accuracy |
|---|---|---:|
| JevBench public 231 tasks | Jev 1.13.0 | 86.58% |
| Same public subset | Open-Jev 2B / 9B / 27B v1.1 | 64.94% / 77.49% / 85.28% |
| JevBench Hard 111 tasks | Jev / Open-Jev 27B v1.1 | 72.97% / 72.07% |
| Typed decisions, 2,000 decisions | Julia 1 | 73.15% |
| Same Julia comparison protocol | Supplied Jev reference | 72.70% |
| MASSIVE 18-scenario classification, 52 locales | Julia 1 | 71.50% |
| Typed decisions, 2,000 decisions | Laya specialist / base English | 76.6% / 36.2% |

Sources: [Open-Jev's JevBench evaluation](https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/jevbench-public.md), [Julia model card](https://huggingface.co/SupersonicLabs/Julia-1), [Laya repository](https://github.com/NandhaKishorM/laya). These are published results, not local reproductions. Open-Jev's report covers only 231 of 534 JevBench tasks and documents differing candidate order between some adapters. Julia's Jev column is a supplied comparison reference, not a fresh matched Jev run. Laya's specialist uses the benchmark's training split, unlike its base zero-shot result. Do not rank all these percentages in one cross-benchmark leaderboard, or compare them numerically with our generated binding test.

TypeSafe's own [workflow evaluation](https://evals.typesafe.ai/) compares workflow decisions against reference labels derived from large-model consensus. Agreement with those references and correctness on independently verified facts are different measurements. Schema validity is also separate from semantic correctness: a well-formed probability object can still assign the highest probability to the wrong candidate.

## Broader generalization takes priority

During this round the user asked to prioritize stable accuracy across domains over a high score on a small controlled corpus. Accordingly, **the 95% diagnostic is not the overall project acceptance criterion**. The original fixed auxiliary-benefit/readiness gates remain in the historical report; they will not be loosened after seeing results to manufacture success.

A separate zero-shot panel was added before opening any final test scores:

- AG News: up to 1,000 deterministic unique test examples from a 2,000-row pool, all four classes.
- DAIR Emotion: up to 1,000 deterministic unique test examples from its 2,000-row test pool, all six classes.
- MASSIVE 1.1: up to 900 shared official test IDs in each of Chinese and English, all 18 scenarios. Parallel translations share groups. Candidate descriptions are fixed human-readable class definitions; they never classify input text.

This introduces three new task families/four task-language cells alongside the existing OCNLI, DuReader-YesNo, and BoolQ public sources. It reports every seed, per-task majority and chance baselines, per-label and balanced accuracy, transferred-temperature NLL/Brier/ECE, and the weakest task. It is a transfer diagnostic: no panel text or labels enter training, checkpoint selection, or new temperature fitting. News/emotion are English-only; this does not establish Chinese performance in those domains or arbitrary-language understanding.

Exact normalized training-state overlap and repeated input text are excluded; over-length inputs are excluded and counted rather than silently truncated. The downloadable dataset-server snapshots are checksum-recorded, not pinned upstream revisions; preserve the ignored raw files for exact reproduction. This is a sampled local protocol, not a reproduction of Julia/Laya/Jev's published protocols.

`--phase all` now includes preparing this panel before training and evaluating it after each primary evaluation. Existing prepared runs can add it separately:

```bash
HTTPS_PROXY=http://127.0.0.1:20122 HTTP_PROXY=http://127.0.0.1:20122 \
  bundle exec ruby benchmarks/decision/generalization.rb --phase download
bundle exec ruby benchmarks/decision/generalization.rb --phase prepare --output runs/decision/evidence-v1
```

The main `evaluate` phase detects the prepared panel and evaluates it in a separate process. Its raw outputs and ASCII summary live in `generalization/`; the main `report.json` includes its three-seed means. This addition does not change the training recipe or the original hypothesis test.


## Shared public benchmarks: completed supplemental evaluation

At the user's request, after the main experiment finished, the unchanged checkpoints were evaluated on [JevBench](https://github.com/fstandhartinger/jevbench) public revision `f8ce71361165846101d02ebc83ad44e47ae44fc3` (231 decisions) and [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) test (400 cases / 2,000 decisions). All three parents and both arms across three seeds are retained. Test labels were never used for training, checkpoint selection, or temperature fitting.

| Full-denominator mean accuracy | Parent | Answer CE | Evidence CE | Input coverage |
|---|---:|---:|---:|---:|
| JevBench public 231 | 19.05% | 19.77% | 19.34% | 58.01% (134/231) |
| Typed Decisions 2,000 | 8.60% | 8.85% | 8.87% | 27.50% (550/2,000) |

The stored 256-state-token / 128-question-option-token limits reject the remaining decisions. They count as wrong in full-denominator accuracy; supported-only accuracy and probability quality are separate diagnostics. These low totals reflect both semantic weakness and input coverage. They are not a matched official leaderboard result: our interface evaluates one question at a time; Typed Decisions' published general-model protocol sends all five together. Criteria descriptions and full label sets are retained, with label prefixes in option text. JevBench ties use lexicographic label order. No private JevBench result, official composite, or speed comparison is claimed.

Typed Decisions gold is teacher-generated agreement, not independently verified truth. Its public training split can be considered separately in a future plan, with soft-label provenance explicit; do not train on these test cases. Raw downloads and prepared records are ignored artifacts, under `data/decision/downloads/shared-benchmarks` and `runs/decision/evidence-v1/shared-benchmarks`.

The shared runner supports `--phase download`, `prepare`, `evaluate`, `report` and `all`. After the primary evidence experiment has completed, reproduce the supplemental panel in that experiment directory (where `shared-benchmarks/` does not yet exist):

```bash
HTTPS_PROXY=http://127.0.0.1:20122 bundle exec ruby benchmarks/decision/shared_benchmarks.rb --phase all --output runs/decision/evidence-v1
```

For the existing completed local panel, use `--phase report`; use `--phase download` to verify cached raw files. Preparation refuses to overwrite an existing panel. Downloads are bounded HTTPS transfers through Faraday. All seven expected raw-file SHA256 values are checked, including cached files. JevBench is revision-pinned; the Typed Decisions dataset-server responses are byte-snapshot-pinned rather than revision-pinned. If that service changes its response bytes, the download fails closed: recover the original cached files or explicitly version a new panel and record its provenance. Do not silently compare a changed snapshot to these results.

Scoring recomputes predictions with lexical tie breaking, rejects invalid/missing-label probability vectors and retains failures in the full denominator and group metrics. Probability quality is calculated only for supported decisions. All nine saved prediction files were rescored during review; their correctness counts and supported counts are unchanged. Regression tests cover schema conversion, full candidate sets, gold/factor isolation, scoring formulas, input-limit failures, cache validation and absolute-path CLI imports.

![Three-seed comparison](../images/decision-evidence-comparison.png)

![Loss components](../images/decision-evidence-loss.png)

In the loss chart, `answer` and `evidence` identify the two experiment arms. The upper panels plot **answer CE** for both arms, with validation NLL unsmoothed; the lower panels plot auxiliary evidence CE. The zero-weight arm does not optimize that auxiliary CE. Falling training loss accompanied by increasing validation NLL explains the early selected checkpoints; it is not evidence of improved generalization.

See the [next experiment protocol](next-experiment.md) for small-set fitting, natural supervised task coverage, independent transfer and context-memory profiling. No next-round training result is claimed.
