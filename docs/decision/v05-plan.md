# Decision v0.5 plan — relational encoding before scale

Status: **planned, implementation not started, 2026-10-03**. The user requested this next-round scope and development plan. This document does not start a development goal or authorize an experiment automatically. v0.4 is committed as `866e59b`; see [v04.md](v04.md) for measured results and [retrospective.md](retrospective.md) for earlier lessons. v0.1 remains the delivered scoring preview.

## Objective and scope

Make the model learn **which actor performed which event**, including explicit negation, before increasing corpus size or model capacity. Compare the current separate encoding with the existing joint encoding on the same reviewed counterfactual core. Verify implementation correctness and positional signals first. If relational learning succeeds, progress to unseen combinations, reviewed natural expressions, missing evidence and calibration under bounded stages.

The intended capability remains one shared EN/ZH neural model: short supplied record + claim/question + candidate descriptions -> candidate probabilities. No mandatory language argument; integer/string IDs and device values remain supported. Ruby constructs JSON. Initially support one or two explicit propositions and single-step support/contradiction/missing-evidence judgments. Four-proposition records, other languages, arbitrary option wording and broad domain competence require their own evidence before being advertised.

An 80% accuracy direction is advisory. A useful release needs independently measured improvement and a clear scope; neither a high aggregate nor a perfect tiny-set fit establishes general semantic understanding. A binary known-evidence preview is an intermediate option, defined before acceptance, with no claim to detect missing evidence.

Exclude external weights, distillation, Qwen preprocessing (removed Path B), RL, dynamic growth, larger width/depth, auxiliary evidence losses and business integration. This round tests representation first. Broader languages and larger corpora remain subsequent work unless the bounded prerequisites below pass.

## Evidence and hypothesis

v0.4's 128-row isolated fit ends at 87.5% accuracy in each language after 1,000 CUDA updates. Different-event cases pass; mixed-actor decisions remain at 50%, with **0% complete mixed-actor groups**. The difficult swaps have identical token-ID multisets but different relations. Full fitting NLL approaches `0.25 * ln(2)`, consistent with the remaining quarter of rows being near chance. Loss/gradient updates, CPU/CUDA parity and inference memory checks passed.

Hypothesis: allowing record and claim tokens to interact throughout the encoder may preserve actor/event/negation relations better than the current separate-encoding/interaction/pooling recipe. This is plausible, not a demonstrated fix. The current network already has positions and cross-attention; do not describe it as missing either feature. Data, optimization and pooling can also influence the outcome.

```text
Reviewed v0.4 core + own v0.1 parent/tokenizer
                       |
Tokenization / masks / positions / gradients / weight transfer
                       |
         +-------------+-------------+
         |                           |
 A: separate encoding         B: joint encoding
 same reviewed fitting rows, seed and optimizer recipe
         +-------------+-------------+
                       |
Training mastery by actor AND event groups
          failure -> diagnosis and stop
                       |
Unseen combinations / wording / record-order development
                       |
Confirm selected representation on a second seed
                       |
Bounded supervised pilot: known -> unknown if mastered
                       |
Freeze checkpoint/profile -> calibration -> fresh acceptance
                       |
Scoped local preview, or documented failed stage
```

## 1. Freeze inputs and verify the computation

Reuse v0.4 reviewed supervision and the exact 128-row fitting diagnostic; its training labels have already been inspected and it is a regression/fitting resource. Preserve all v0.4 artifacts. Do not continue its optimizer or use its fitted weights as a parent. Start each arm from our own delivered v0.1 parent/tokenizer, with a fresh optimizer.

Before training, record rendered inputs, token IDs, segment boundaries, masks and positional indices for representative actor swaps in both languages. Check:

- Opposite-gold actor swaps have the same token multiset but different ordered sequences; neither actor's clause nor the claim is truncated or normalized away.
- Separate sequences have their intended positional indices; joint positions span record, claim and option, with explicit separators. Padding never becomes evidence, and candidate pooling selects the actual answer tokens when used.
- Position tensors affect the encoder in a deterministic synthetic test. A same-multiset sequence-order diagnostic with positions enabled/disabled distinguishes the mechanism from an accidental order-insensitive pooled path. Do not demand semantic correctness from untrained weights or infer understanding from any nonzero logit difference.
- All encoder/scoring parameters are registered, updated and preserved across validation. Check representative effective gradients and an optimizer step, not only finite final loss.
- Candidate permutation/chunking, mixed candidate counts, numeric/string IDs, valid normalized JSON and deterministic checkpoint resume work for both encoding modes.

Fix demonstrated implementation bugs before freezing the comparison. If a bug changes the control, label the new control accurately and preserve the old reference. Freeze code revision, config, parent/tokenizer/data hashes, parameter transfer and evaluation rules before running either arm.

## 2. Compare two bounded representation recipes

| Factor | A: control | B: candidate |
| --- | --- | --- |
| Encoding | Current separate record and question/option encoding, then interaction | Existing joint record/question/option encoding |
| Position encoding | Current sinusoidal settings | Same sinusoidal settings |
| Width, encoder depth, vocabulary | Parent settings | Same parent settings |
| Pooling / scoring | Parent settings | Same settings where supported |
| Training | Ordinary candidate CE | Same |
| Fitting data / sampling | Frozen 128 rows, complete balanced groups | Identical rows and effective batch schedule |

The code already exposes `encoding_mode: joint`. This is not a pure one-variable parameter-matched architecture experiment: joint mode omits separate interaction/matching modules and repeats record tokens per candidate. Report active parameter counts, valid/padded encoder tokens, updates, examples, elapsed time and peak memory for each arm. Claim a **representation-recipe benefit**, not proof that joint attention alone caused it.

Transfer shared encoder and compatible score tensors by explicit key/shape mapping, verifying copied values against the parent. List missing, unused and newly initialized tensors; seed any new parameters deterministically. Never silently accept a partial checkpoint load or reinterpret tokenizer IDs. Shared initial tensors must match exactly. Do not invent compatibility scaffolding solely to retain older formats; forward iteration is preferred.

Primary fit seed 1337; max **1,000 updates per arm**, evaluation/checkpoint every 100, LR `0.0003`, no warmup, dropout zero, FP32, effective batch 32. Use the v0.4 fitting sampler/order and plain CE. CUDA first, 4,096 MiB process budget on the 6 GiB Quadro RTX 3000, supported CPU fallback. Joint mode may need a smaller microbatch: preserve effective examples/gradients, record the adjustment and do not claim equal GPU compute from equal updates.

Fit mastery requires, in **each language**, at least 99% balanced known accuracy and 90% complete mixed-actor **and** mixed-event group correctness, for two consecutive checks. Select the earliest qualifying checkpoint; otherwise report the final checkpoint and failure slices. These are engineering learning checks on a repeatedly visited small set, not publication accuracy.

Report all arms, including failures. Positional removal is an evaluation-only diagnostic on training/development material, not a third trainable arm. No rotary/position-scale/pooling/learning-rate sweep in this round. If neither recipe learns the core, stop before expanding data, unknown training or calibration; document the strongest remaining causal hypotheses and propose a separately bounded intervention.

## 3. Check composition before scaling supervision

The tiny fitted checkpoints may be evaluated on development material only; they never initialize the full supervised pilots. Separate the following abilities:

1. Familiar vocabulary with unseen actor/event combinations.
2. New actor identities and events, with tokenizer coverage reported.
3. Reversed record order and irrelevant facts, with invariance scored by correct gold groups.
4. Held-out state/claim wording and negative assertions.
5. Held-out candidate descriptions, reported separately from canonical options.

Use complete bilingual families and report actor/event groups separately. Keep class-balanced accuracy, per-class recall, NLL, confusion matrices and margins alongside complete-group correctness. A flipped output can still be wrong; count groups only when all members are correct. Invariant constant outputs are negative metric fixtures.

Reuse v0.4 development only as **observed development**, never fresh acceptance. Freeze a v0.5 split/source-history manifest before pilots. Existing v0.4 acceptance material can be retained only if the audit confirms no predictions or training/selection exposure and no overlapping material; otherwise reserve new families. An unopened marker alone is insufficient. Review/label qualification before freeze is disclosed, distinct from checkpoint evaluation. Related variants/translations stay together.

Rank fitting-qualified arms by worst-language canonical development balanced accuracy, then worst actor/event group correctness, then NLL; prefer the simpler recipe for an exact tie. If only one qualifies, make no held-out paired architecture-win claim. Repeat the selected fitting recipe with seed 2027, max 1,000 updates, applying the same mastery checks. No extra seed or budget extension if confirmation fails. Poor unseen transfer despite repeatable fitting is a reason to test supervised generalization in the next stage, not to claim the tiny fit solved semantics.

## 4. Conditional supervised generalization and unknown stage

Only a repeatably fitting representation enters this stage. Initialize from the original own parent through the recorded transfer, not fitting weights. Train the selected recipe with the reviewed v0.4 controlled core, qualified natural examples and routing replay. Keep labels/canonical profiles and complete-group sampling. Reuse the staged trainer where correct; avoid duplicating the existing audit/sampler/evaluation machinery.

Primary pilot seed 1337, max **2,000 updates**, known stage at most 1,000, check every 100. Defaults: LR `0.0001`, warmup 100, dropout 0.1, FP32, effective batch 32. Preserve v0.4's 60% controlled / 20% natural / 20% routing exposure cycle and equal EN/ZH allocation. Freeze actual eligible counts and visits; qualify new natural examples before this pilot, not in response to acceptance errors.

Aim for 200 reviewed in-scope natural decisions per language with source/component isolation. This is a coverage goal, not permission to invent or relabel ambiguous examples. Existing qualified data is small. Expand only publicly available, licensed sources with target-blind record/claim review; retain exclusions and reviewer provenance. If the goal is not reached, disclose the shortfall and avoid claiming broad natural-language competence. No claim of independent human agreement from one assistant review.

Known-stage advancement: at least 95% balanced training-probe accuracy, 90% complete mixed-actor and mixed-event groups, and at least 55% canonical development balanced accuracy in each language for two consecutive checks. Preserve a pre-unknown checkpoint. Failure stops the stage; unknown supervision cannot repair an unlearned known relation by being introduced early.

Then add support/contradiction/insufficient-evidence cases, including absent actor and present actor/unrecorded event, while replaying known groups. Explicit negative evidence is not unknown; conditional `depends` is not unknown. Report known recall and actor binding throughout. If the joint three-state stage loses known competence, retain the eligible known-only checkpoint and document forgetting.

Development eligibility for a scoped preview:

- **Binary known-evidence:** per-language balanced accuracy ≥75%, complete mixed-actor and mixed-event groups each ≥50%.
- **Three-state:** per-language recall for each of the three classes ≥70%, complete mixed-actor and mixed-event groups each ≥50%.
- For either profile, improvement over the unchanged parent must be at least five percentage points in each language on its primary balanced measure; fresh in-scope natural accuracy ≥65% and no worse than parent by more than three points. Report domain/source counts and uncertainty; small subsets cannot establish domain stability.

These are minimum eligibility checks, not the desired endpoint. Aim toward 80% or better while improving difficult groups. Freeze them before pilots; do not lower them after observing acceptance. If no checkpoint qualifies, stop without calibration or acceptance. If one qualifies, confirm the full selected recipe with seed 2027, max 2,000 updates, and report both seeds. This round allows **at most 7,000 optimizer updates total**: two primary fits, one fit confirmation, one pilot and one pilot confirmation. No unused budget is reassigned to extend a failed stage.

## 5. Calibration, acceptance and local delivery

Only independently confirmed development-eligible profiles enter publication. Freeze the checkpoint, capability contract, candidate wording and selection policy before calibration. Fit temperature on separate calibration families; report raw/calibrated NLL, Brier score and reliability bins with counts. Two- and three-choice profiles may need different calibration policies, selected by profile rather than a mandatory language flag. Temperature does not fix incorrect rankings.

Open reserved acceptance once, recording protocol/checkpoint hashes and an evaluation marker. Compare unchanged parent and the selected checkpoint on the same frozen records. Report per-language/class/domain scores, actor/event complete groups, held-expression views and family-level uncertainty. Reusing views of the same frame does not multiply independent sample size. Apply the same declared eligibility floors to acceptance; do not select another checkpoint or rescue thresholds using test results. Failed acceptance stays a measured failure.

Package a factual preview only for the passing profile. A binary profile returns probabilities conditional on supplied choices and cannot infer answerability. Unknown probabilities indicate insufficient evidence under the learned record contract, not general epistemic certainty. Keep v0.1 routing available separately unless routing replacement is independently supported.

Verify warm/cold CPU and CUDA latency, representative batches and candidate counts, truncation rejection, permutation/chunking consistency and memory over repeated inference and validation. Measure actual joint-path caching: it currently re-encodes the record with each option, so do not promise reusable record-prefix embeddings. Optimize only measured overhead after correctness, preserving probabilities. No quantization or speed target is necessary to explain away a failed semantic result.

## Implementation locations and required tests

```text
lib/easy_ai/decision/
|-- choice_model.rb / encoder.rb / data/collator.rb  # verified encoding paths
|-- judgment_trainer.rb / data/judgment_sampler.rb   # reused stages and groups
`-- checkpoint.rb                                  # inspect actual loader/transfer API
benchmarks/decision/
|-- relational_encoding.rb                         # proposed bounded v0.5 runner
`-- relational_encoding/                           # split helpers only as needed
test/easy_ai/decision/
`-- relational_encoding_test.rb                    # proposed meaningful regressions
docs/decision/
|-- v05-plan.md                                    # this plan
`-- v05.md                                         # actual results and limitations
runs/decision/relational-v05/                       # ignored protocol/data/weights/logs
docs/images/decision-v05-relational.png             # one reviewed results figure
```

Confirm existing loader filenames/APIs during implementation; these paths describe responsibilities, not a reason to create a redundant subsystem. Add a minimal explicit weight-transfer helper only if the current loader cannot perform and validate this operation safely.

Necessary tests: ordered token/mask/answer-span correctness; padding invariance; positional-signal mechanism; verified shared-weight transfer and rejected shape/vocabulary mismatches; active gradients; effective accumulation for joint/separate modes; live parameters and deterministic resume; actor-blind negative metric fixtures; audit/split integrity; stage and acceptance guards; inference permutation/chunking/IDs/JSON; CPU/CUDA parity and memory. Reuse passing existing tests where sufficient. Fix implementation errors before attributing failure to the model.

Checks: `bundle exec rake test`, `bundle exec rake test:learning`, `bundle exec rake lint`. Keep the learning examples/figures intact. Downloaded data, reviews, weights, caches and raw logs stay ignored. Track one readable chart derived from observed values, with separate training and held-out curves, actor/event slices and no invented smoothing trend.

Deliver the bounded runner, tests, reproducible protocol and parameter-transfer manifest, all executed results and stopped-stage reasons, plots and loading examples. Update README, handover and retrospective with successful and failed lessons. A completed development round can have no promoted checkpoint; its completion must not be described as achieving a useful factual model unless independent evidence supports that claim.

## Outcome decisions

| Observation | Action |
| --- | --- |
| Token/mask/position/transfer bug | Fix, test, then freeze a corrected comparison |
| Both encodings fail actor fitting | Stop; no larger mixture or calibration; propose a separate intervention |
| Joint fits, separate fails | Confirm; attribute only a recipe-level fitting benefit |
| Both fit but unseen relations fail | Diagnose composition/coverage under the bounded supervised pilot |
| Canonical transfer passes, wording/options fail | Keep prospectively scoped canonical capability; record uncovered expressions |
| Unknown learning destroys known judgments | Preserve qualified binary checkpoint; no claim of answerability |
| Development/confirmation fails | Stop before acceptance; report the failed stage |
| Confirmed model passes fresh acceptance | Deliver measured local factual preview within its validated scope |

The next optimization after this round depends on these results. Data/model scaling becomes a justified candidate only after the existing relational core is learnable and the remaining error is demonstrably held-out transfer or capacity-related.
