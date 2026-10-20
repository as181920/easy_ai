# Decision v0.8 — fresh evaluation and event/expression generalization

Status: **goal authorized on 2026-10-20; prospective protocol before predictions/training**. The user requested completing this round after adopting fresh evaluation and event/expression coverage as the next direction following the [completed v0.7 comparison](v07.md). Dataset catalogs, source manifests and the executable protocol must be frozen before prediction or training; no v0.8 measured results exist at protocol preparation.

The question is whether the repeatable own-model actor-binding recipe transfers to fresh worlds and can learn better event/expression generalization while retaining actor binding. Keep the 6,627,841-parameter separate/RoPE architecture, own tokenizer, Ruby/Torch.rb stack and CUDA training. Model scaling, a learning-rate sweep, unknown supervision and a new release are separate later questions.

Isolation correction before replacement-panel predictions: the first data draft is retired after a cross-template augmented-family audit discovers overlapping canonical facts. Preserve its protocol, data, predictions and traces under `runs/decision/event-generalization-v08/`. Replay actual renaming for both v0.7 parents and all executed v0.8 prefixes, validate exact agreement with saved prefix surface hashes, and exclude those augmented families/materials as well as historical panels when drawing replacement worlds. Refresh the novel-name pool so retired-draft identities are not presented as newly reserved identities. The corrected root is `runs/decision/event-generalization-v08-corrected/`. Keep event meanings, template partitions, world allocations, training recipes, seeds and selection rules unchanged. Replacement generation uses only isolation evidence, not scores; disclose that this repair occurs after observations on the retired draft. No replacement-panel prediction precedes its protocol freeze.

Retain committed step-1,500 model/optimizer/sampler prefixes only after proving zero overlap with replacement worlds. A sealed migration binds previous/current exclusion signatures and checkpoint weights; strengthened exclusions reject every rendering form of a reserved family. Preserve the old exclusions too, so earlier rejected draws remain rejected. The discarded tails require 151 replayed updates, counted with inherited feasibility once. A new shape preflight may add another 12 updates; the original 37,000 ceiling remains unchanged. Old development curves are retired rather than mixed into replacement-panel curves, and no additional model search is authorized by this correction.

## Baselines and initialization

Preserve both v0.7 combined endpoints, seeds 1337 and 2027, and every earlier success/failure. Each v0.8 arm starts from the corresponding seed's exact v0.7 combined weights, with a new AdamW optimizer and explicit lineage. It is a new training stage, not an exact resume of an unchanged objective. Reset optimizer state identically for reference and interventions; retain LR .0003, zero dropout/warmup, effective batch 32, microbatch 16 and two-way accumulation.

Keep the complete original 8,192-row v0.7 training corpus and balanced four-query sampler. All arms retain person renaming and actor-pair loss. The unmodified v0.7 endpoints are read-only baselines; a matched continuation control distinguishes intervention gains from simply training the existing recipe longer. The old v0.6/v0.7 development panels become regression-only and never count as fresh evidence.

## New data and evaluation separation

Create a versioned catalog of additional bilingual events and meaning-preserving surface forms. Do not relabel the four previously observed held-out events as fresh. Assign explicit event IDs to training, development-only and lockbox-only pools before rendering. Separate surface-template IDs into training, development-only and lockbox-only pools. Event novelty and wording novelty are different axes: held-out expression views reuse trained event meanings, whereas held-out event views use disjoint event meanings. Keep claims unambiguous and binary; absent facts/unknown labels are outside this round.

Reserve world families before generating truth assignments, record orders, claim polarities or paraphrases. Each panel balances actor and event worlds and includes complete counterfactual truth assignments and complete query units. No family crosses train/development/lockbox. Reject exact and normalized evaluation-material overlap with historical local panels and augmented training outputs. Repeated renderings of a family do not create additional independent worlds. Novel-person pools must be excluded from renaming; report the precise audited training-material scope of the novelty claim.

Freeze five separate views in each language:

| View | Primary change |
| --- | --- |
| New combinations | Fresh combinations of training-pool identities and events, canonical wording |
| New actors | Held-out identities, training-pool events, canonical wording |
| New events | Held-out event IDs, training-pool identities, canonical wording |
| New expressions | Held-out surface templates for trained event meanings |
| Composed shift | Held-out identities, events and expressions together; stress diagnostic |

Target **32 independent worlds per language/view for development** and **64 for the lockbox**, balanced between actor and event worlds. With 32 binary rows per world (including both record orders), these are 10,240 and 20,480 rows across five views/two languages. Each axis has **128 development or 256 lockbox mixed-truth full pairs per language/view**, sharing 16 or 32 worlds respectively. This corrects the planning denominator before any predictions: the initial pair count omitted the second record order; row and world allocations are unchanged. Report both denominators. Reject finite-catalog exhaustion rather than repeat worlds or silently reduce the panel.

Gold follows stored facts, but metadata is never a neural inference feature. Independently review explicit bilingual positive/negative examples and validate renderer/oracle agreement. A synthetic template catalog is not independently collected natural-language QA. Freeze catalog, exclusions, source/data/parent/tokenizer hashes and protocol before opening development predictions. Evaluate the existing two v0.7 baselines on fresh development first; once observed, that panel is development rather than acceptance.

The lockbox is generated and audited before training, with **no model predictions until recipe and confirmation checkpoints are fixed**. Use different families, novel identities, held-out event IDs and held-out template IDs from development. When confirmation is triggered, final evaluation compares fixed original v0.7 baselines, v0.8 continuation references and the selected candidate for both seeds. Otherwise, compare baseline/reference only at seed 1337; the original seed-2027 baseline may be reported descriptively without inventing a trained counterpart. Open it once, preserve unfavorable outcomes and do no further tuning on it. This is a controlled synthetic transfer check, not broad production acceptance.

## Four-arm comparison

Run the full coverage × event-loss factorial at seed 1337, each for exactly **6,000 additional updates**:

| Arm | New event/expression coverage | Event-pair auxiliary loss |
| --- | --- | --- |
| Continuation reference | No | No |
| Coverage | Yes | No |
| Event loss | No | Yes |
| Combined | Yes | Yes |

Coverage is training-only augmentation of the sampled original world units, not a change to the sampler. In each language/axis bucket, render one of the two sampled complete units with events/forms from the frozen training pools and retain the other's canonical event/expression rendering. Draw randomness independently from sampling, candidate permutation and person renaming. Preserve truth, assertion polarity, record order and complete pairing; reject reserved material. All variants therefore have identical original-row visit arrays. Separately record actual event/template frequencies and augmented-surface coverage; original-row visits are not counts of unique rendered examples or equal convergence.

The event-loss arms retain CE and the v0.7 actor term and add `.5 * mean softplus(1 - (z_supported - z_contradicted))` over complete opposite-label **event** query pairs, with `z = logit_yes - logit_no`. Validate pair membership and candidate IDs after permutation. Skip same-label pairs. Report CE, actor and event terms separately, plus gradient norms and clipping. The intervention changes relative weighting as well as pairing; it does not isolate a unique coupling mechanism.

Measure a fixed original-training probe and a fixed augmented-training probe every 500 updates. Use a prospectively fixed small development subset for learning curves; evaluate all development views at the final endpoint. Record maximum actual token lengths, truncation count, per-process memory and actual CUDA device. Retain existing token caps initially (256 state / 128 question-option tokens), with truncation errors. Shape/memory preflight must cover the new corpus's actual maximum lengths before any long run; amend the prospective configuration if necessary, rather than silently truncate or change accumulation mid-comparison.

## Direction, confirmation and budget

The primary directional target is the minimum event full-pair correctness across **new-event and new-expression development views, in both languages**. Require at least a five-percentage-point gain over the matched continuation reference. Also require no more than three-point balanced-accuracy loss in any of the four basic development views/languages and no more than five-point loss in the worst actor full-pair cell across those views and the old regression panels. Rank qualifying arms by primary event score, then worst actor score, then worst balanced accuracy and NLL. Report every view, row/pair denominator and regression; composed shift remains a stress diagnostic rather than an initial all-or-nothing gate.

The 95% training balanced / 85% actor-and-event group / two-consecutive-probe checks remain diagnostics. A miss does not stop the fixed experiment or imply an incorrect architecture. Report meaningful partial gains even if the declared full directional criterion fails. Retention and production eligibility are different judgments.

If a candidate qualifies, repeat it and its matched continuation reference at seed 2027 for 6,000 updates each. Apply the same rule without changing thresholds. Fix these checkpoints before the single lockbox comparison. If no primary candidate qualifies, complete the negative development conclusion and evaluate only the prospectively defined baseline/reference comparison on the lockbox; do not nominate a favorable checkpoint retrospectively. A subsequent pretrained-encoder diagnostic would need a separate frozen protocol.

Maximum main training: **36,000 additional CUDA optimizer updates** (24k factorial + 12k conditional confirmation). Hard experiment ceiling: **37,000 actual updates**, including feasibility and replay. Read-only evaluation adds no optimizer updates. No automatic extensions, model-size search, learning-rate sweep or spending unused budget on more seeds. Use a 4,096 MiB per-process soft budget and reject silent CPU fallback for comparable training. CPU remains appropriate for deterministic tests and independent inference audits.

## Required implementation and completion evidence

- Use a separate v0.8 namespace/entrypoint and new output root. Preserve historical benchmarks and frozen sources. Support string/symbol enum keyword values.
- Test explicit gold examples, bilingual renderings, truth/polarity/order preservation, template/event/family separation, reserved-material rejection, complete units, candidate-ID alignment, actor/event loss values and gradient signs, matched sampling, optimizer resume and protocol/result tampering. Assert deterministic logic, not convergence or weight distributions.
- Validate all split data and maximum lengths before learning. Keep all failed probes/invocations, source amendments and replay updates. Protect existing outputs and use per-arm locks.
- Finish every registered primary arm, conditional confirmation and final lockbox evaluation. Audit independent pair counts, public candidate order/chunking and CPU/CUDA behavior with a fixed stratified subset; retain full GPU raw predictions for every final view. Report the subset's exact scope rather than claim full CPU equivalence.
- Deliver measured results with counts, paired regressions and seed variability; charts, resource/update ledger, machine-readable summaries, loading/reproduction instructions and updated README/handover/retrospective. Distinguish baseline transfer, continuation gains and intervention gains. The result may be positive, partial or bounded negative. v0.1 remains delivered unless a later separately scoped release is validated.
