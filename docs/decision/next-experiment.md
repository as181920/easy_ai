# Next experiment: fitting, natural-task coverage and transfer

Status: protocol proposed on 2026-10-01; **not yet trained**. The completed evidence round is documented in [evidence.md](evidence.md). All previous tests are now observed diagnostics, not fresh acceptance sets. Continue with our own randomly initialized/trained weights, Ruby/Torch.rb and GPU-first execution. Do not introduce pretrained weights, teacher inference, RL or linguistic keyword rules in this round.

## What the results establish

| Observation | Implication | What it does not establish |
|---|---|---|
| 128 training rows reach only about 75% answer accuracy after 300 updates | Small-set fitting must be investigated before scaling | A model-capacity limit or a convergence guarantee |
| Sentence evidence improves, answer accuracy does not | The auxiliary objective is not sufficient under this recipe | That evidence supervision can never help |
| All six best validation checkpoints are step 100; training loss keeps falling | Longer training under this mixture fails to improve held-out NLL | A universal optimal training length |
| AG News, Emotion and MASSIVE transfer is weak | Current task coverage does not support broad semantic claims | That adding these tasks will solve unseen workflows |
| JevBench coverage 58%; Typed Decisions coverage 27.5% | Context/candidate limits cause many failures | That all supported cases are understood |
| Supported-only answer-arm means are about 34.08% / 32.18% | Fixing context coverage alone is insufficient | A directly comparable official leaderboard score |

Supported-only figures divide full accuracy by coverage, which is identical across seeds here. Shared-benchmark results remain evaluation-only. Typed Decisions references reflect teacher agreement; they are not independently verified truth.

```text
small-set fitting audit
        |
        +-- cannot fit --> inspect gradients, alignment, scorer and optimization
        |                  change one factor, retain failure traces
        |
        +-- can fit ----> broader natural supervised training
                              |
                              +--> held-out examples of trained tasks
                              +--> held-out task family / domain transfer
                              +--> old binding and public panels: regression only

context coverage / CUDA memory profiling: separate engineering measurement
model-size increase: deferred until fitting and data controls are resolved
```

## Stage 1: identify the small-set bottleneck

Use the existing 128-row sanity set; it is training data, not a generalization benchmark. Keep the full candidate set, shuffle candidate order, and measure both training accuracy and correctness of complete fact-flip/binding groups. Log answer CE, gradient norms by embedding/encoder/interaction/scorer, and state sensitivity when state rows are shuffled. Check that different states really produce different input tensors and nonconstant logits.

Start from the current parent with answer CE only and the existing fixed schedule. Extend the diagnostic budget to 2,000 updates, evaluating the entire tiny set every 100 updates. Use dropout zero for this memorization diagnostic. Record failures rather than adjusting labels. At least 99% training accuracy and 95% training fact-flip correctness are diagnostic targets only. If they fail, inspect persistent cases and compare a randomly initialized model with the same configuration/budget; this distinguishes a poor starting point from a failure shared by both starts. Do not treat either outcome as a fresh-test result.

If both starts still fail, inspect the learning-rate schedule and the matching scorer with one controlled change at a time. This is where an interaction/scorer revision becomes justified; adding layers before these checks would obscure the cause. A diagnostic target is a stop-and-investigate rule, not a promised achievable result.

## Stage 2: expand actual task coverage

After fitting is demonstrated, prepare full-candidate supervised training from official **training** splits of AG News, Emotion and MASSIVE en-US/zh-CN, alongside current BoolQ, DuReader and OCNLI sources. Use human/dataset labels for this round. Leave Typed Decisions training out initially so teacher agreement cannot be confused with gold semantic supervision. Do not convert test rows into training data.

Before training, create an immutable manifest with source revision/checksum, label definitions, language, task, normalized state hashes, group IDs and all split counts. Reserve disjoint validation, calibration and held-out evaluation groups from the new training sources. Use MASSIVE original IDs to keep parallel translations in the same split. Reject overlap with any existing evaluation panel. Verify dataset licenses/provenance and missing/truncated fields during preparation. Fixed human-written candidate definitions describe labels; they do not classify input text.

Run two paired conditions using the same accepted starting checkpoint, architecture, optimizer, candidate sets and seed schedules:

- Control: previous public-source mixture, answer CE only.
- Broad: new six-task mixture, answer CE only, source-balanced sampling.

For each, use seeds 1337/2027/3407, 4,000 updates, effective batch 32, microbatch chosen from actual CUDA memory profiling, learning rate 1e-4, warmup 100, validation every 200 updates. These are initial protocol values, not tuned results. Fix them before exposing new held-out labels. Select checkpoints with **the same common validation panel**, reporting per-task NLL as well as equal-task macro NLL; do not compare models selected against different panels. Fit one global temperature on the independent calibration split afterward.

Include a leave-one-task-family-out diagnostic: withhold a declared natural task family from training and checkpoint selection, then evaluate it after all paired fits complete. Report this separately from improvements on trained tasks. The initial small-set investigation is deliberately adaptive; freeze this larger experiment only once that investigation is finished.

## Stage 3: context support without hidden truncation

Measure tokenizer length distributions from training/validation inputs first. On the same checkpoint, profile inference-only limits at state/candidate 256/128, 512/256 and 1,024/512 tokens, with microbatch 1–8 as memory permits. These models use sinusoidal positions, but extrapolation can still hurt accuracy. Keep each setting as a separately named measurement; never replace the historical 256/128 benchmark score.

Measure live/peak CUDA memory across repeated validation passes, not just boundary process memory. Preserve CPU fallback, record the actual device and latency, and compare coverage, supported-only accuracy and full-denominator accuracy separately. Increasing inference limits does not provide long-context training. If a limit becomes part of the training recipe, rerun both paired arms under that same limit before attributing a difference to data coverage.

## Evidence required to adopt a change

The overall objective is reproducible cross-domain progress, not a global 95% score. For the paired natural-task round, preregister these practical adoption criteria:

- Equal-task macro accuracy improves by at least 3 percentage points on the new held-out evaluation; gains appear in at least two of three seeds.
- Report every task/language cell, chance, majority baseline and paired seed differences. Investigate any cell regression exceeding 3 points rather than averaging it away.
- Calibration NLL/Brier and the held-out-family result accompany accuracy. Improvements confined to newly trained task families count as specialization, not broad transfer.
- Binding/fact-flip correctness, state sensitivity and input coverage remain visible. JSON validity is an interface property and does not demonstrate semantic correctness.

These thresholds are proposed engineering decisions, not estimates of the probability of success. Three seeds provide limited uncertainty evidence. Freeze exact split hashes and acceptance rules before evaluation; document deviations. Do not promote a checkpoint merely because it passes one aggregate gate.

## Handover sequence

1. Instrument and run Stage 1; record the first persistent failure and tested remedy.
2. Freeze Stage 2 dataset manifests and common validation/calibration/evaluation partitions.
3. Profile Stage 3 memory before choosing the fixed training context/microbatch.
4. Execute paired runs sequentially on CUDA, with logs/charts and checkpoint hashes.
5. Evaluate once, record success and failure in the retrospective, and decide whether architecture or scale is the next bottleneck.
