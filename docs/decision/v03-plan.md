# Decision v0.3 plan — robust factual scoring with own weights

Status: **Completed, no promotion, 2026-10-03.** The user first requested planning only, then explicitly authorized completing this plan as a new goal. Implementation, audits, corrected fitting, both 2,000-update CUDA pilots, calibration, one-time fresh evaluation, runtime verification and handover are complete; no eligible checkpoint triggered confirmation. Read [v03.md](v03.md) for measured results and disclosed corrections. v0.1 remains delivered; v0.2/v0.3 weights remain diagnostic. The plan below records the intended procedure, not an accuracy claim or authorization for another sweep.

## Intended outcome and scope

Produce a useful shared Decision model whose factual judgments survive changes in queried actor, fact order, negation and equivalent candidate descriptions, while retaining the delivered routing capability. Improve natural-text transfer rather than only finite-template accuracy. Continue with our own trained weights, Ruby/Torch.rb and one shared multilingual inference API. No business integration.

Chinese and English are the measured languages for this round. UTF-8 input support is not evidence of additional-language understanding; add languages later with supervised coverage and independent evaluation. Candidate IDs accept numbers or strings; JSON keys are strings. Inference needs no language argument. Probability JSON remains assembled by Ruby; JSON generation is not a training objective.

The user treats 80% as an advisory direction, not a universal release gate. Success means reproducible useful improvement without hiding catastrophic slices or unacceptable capability loss. This plan does not promise general semantics, an accuracy percentage or a probability of success.

## Evidence motivating the work

Read [v02.md](v02.md) and the historical [retrospective](retrospective.md) before implementing.

| v0.2 CE result | Observation | Next diagnostic |
| --- | --- | --- |
| Factual macro: 46.33% → 67.33% against v0.1 | Task supervision has useful effects | Preserve CE; improve what is supervised |
| English actor-binding complete pairs: 0.31% | Binding remains nearly unsolved | Independently vary actor, query and fact order |
| Chinese unknown accuracy: 0% | Missing evidence is not handled reliably | Audit targets, candidate semantics and actual exposure |
| Chinese complete pairs: 99.92% | Finite-world templates are learned | Require unseen expressions and natural-text transfer |
| Natural QA/NLI EN/ZH: 63.50% / 55.51% | No improvement over own broad parent | Report public sources separately from synthetic data |
| Routing EN/ZH: 68.38% / 70.40% | Below v0.1 regression reference 78.43% / 81.09% | Start from delivered own parent and track retention |
| Alternative candidate wording: 31.25% on 48 training diagnostic rows | Strong expression sensitivity | Review semantic equivalence; isolate wording effects |

These are observations, not causal proofs. In particular, lower Chinese unknown exposure is not a demonstrated explanation of its collapse. The broad initializer started with weak routing, so final routing loss cannot all be attributed to forgetting.

Earlier candidate-wording expansion reduced accuracy in the coverage round; grouped actor training and curriculum also had mixed seed results. This round must correct concrete correlations and measure transfer, rather than repeat those ideas under a new name. Pair-margin had no demonstrated advantage in v0.2 and is not the default.

## Planned procedure

```text
Audit historical exposure, parent weights and failure slices
                         |
Record explicit worlds and independently vary semantic factors
                         |
Freeze train / development / calibration / fresh acceptance
                         |
Verify gold, tokenization, fitting and unseen-combination behavior
                         |
Own v0.1 parent + ordinary CE + routing replay
          |                                  |
Current-recipe control                Corrected-data candidate
          +-------------------+--------------+
                              |
          Development selection with retention checks
                              |
             Conditional second-seed confirmation
                              |
           Independent calibration and final evaluation
                              |
            Publish measured preview OR retain v0.1
```

### 1. Audit and freeze the experiment

- Locate the exact own v0.1 source checkpoint and published weights. Check architecture, tokenizer IDs, tensor shapes, parent SHA and loader behavior before initializing training; do not silently substitute the broad parent or change vocabulary.
- Reproduce binding, unknown and wording failures on development/regression material. Check distinct tokenization, labels, attention masks, gradient flow, real parameter updates and option-order invariance. No inference keyword rules.
- Audit all historical public-data components, translations and generated families before assigning new splits. v0.2 acceptance, old lateness examples and previously opened panels are regression-only.
- Freeze a protocol before pilot training: source versions/licensing, split and tensor hashes, exposure schedule, training budget, checkpoint selection, retention tolerance, confirmation rule and release decision. Any adjustment after inspection must be disclosed as development, not retroactively called blind acceptance.

### 2. Build supervision that removes known correlations

Represent each generated world explicitly and derive targets from its facts. Independently vary actor identity, queried actor, event, fact position/order, polarity, question polarity, irrelevant facts and candidate order. Include mixed truth values and same truth values; actor changes need not always flip the answer.

```text
State: Lin bought a ticket. Zhou did not buy a ticket.

Query Lin bought?            -> yes
Query Zhou bought?           -> no
Query Zhou did not buy?      -> yes
Query Wang bought?           -> insufficient information

Reversing the two facts preserves these answers.
```

Use same-state question changes alongside fact changes. Training/evaluation group metadata must distinguish answer-flipping and answer-preserving transformations, and must never enter neural input. Group-level correctness requires all members correct; constant predictions cannot pass merely by remaining invariant.

Unknown means absence of relevant evidence in these explicit worlds, not negative evidence. Preserve the distinct annotated meanings of public neutral/depends examples; do not flatten modality, conditionality or conflicting evidence into false. Review question/candidate semantics where these categories differ.

Audit supervision jointly by language, known/unknown target, phenomenon, queried role, fact position and source. Balance declared exposure where appropriate and log actual unique groups, visits and valid tokens. Equal language sampling alone is insufficient. Keep replay exposure explicit.

Add reviewed natural paraphrases and eligible public gold QA/NLI, not just more names in identical templates. Candidate variants must have equivalent meaning for the particular question. Preserve support/contradiction/insufficient-information distinctions and train multiple candidate counts. Two-option output remains conditional on supplied options; it cannot represent a missing unknown alternative. Reserve unseen candidate-expression families for evaluation.

### 3. Reserve evaluation before training

Split complete semantic worlds, translations, paraphrases and source components together. Separately reserve unseen expression families and event/domain families; unseen random IDs are not sufficient. Exclude whole historically exposed public components, including different answers to the same question.

Keep three clearly reported panels:

1. Controlled fresh groups for actor binding, fact/question flips, order invariance, irrelevant facts, unknown and candidate wording.
2. Fresh natural public QA/NLI and reviewed natural-expression material across multiple domains.
3. Previously observed routing/factual regressions, explicitly marked as such.

Target a few hundred independent controlled families per language and a few hundred eligible natural examples per language, subject to the provenance audit. Freeze actual counts and source/language cells before fitting; document shortfalls rather than silently relaxing independence. Report group-bootstrap intervals and worst cells. Use training/development diagnostics for state/question removal; do not treat unchanged original targets on altered inputs as new gold judgments.

### 4. Fit, then run one bounded comparison

Use a 128–256-example representative training fitting check, including binding, unknown, same-state question changes and candidate descriptions. Near-perfect fitting is a debugging direction, not generalization evidence. Separately evaluate development combinations/expressions absent from the fitting set. If fitting fails, stop the pilot and inspect implementation/optimization first.

Keep ordinary candidate CE, the existing model structure and tokenizer. No margin, new evidence head, MLM phase, dynamic growth, teacher supervision or RL in this round. Continue routing replay; use the own delivered v0.1 parent as the common initialization.

Planned bounded comparison:

- **Control:** frozen v0.2 data/sampling recipe, initialized from the same v0.1 parent. All its historically exposed cases are development/regression material.
- **Candidate:** corrected factorization, reviewed candidate/natural expressions and declared per-language unknown exposure, with the same parent, optimizer, effective batch, replay allocation and maximum update budget.

This compares a supervision package; it cannot attribute improvement to any single component. Do not claim matched examples/tokens when corrected data changes sequence lengths or eligible sampling. Record both and disclose differences. Check new acceptance against both arms for historical overlap before freezing it.

Planning bounds: 20k–40k training rows per arm, at most 2,000 updates per arm, effective batch 32, validation every 100 updates, seed 1337. Fix the LR schedule after the fitting check and before the pilot. Keep within the existing 4,096 MiB GPU process budget on the 6 GiB RTX 3000; CUDA first with supported CPU fallback. Extra steps, data downloads and a wider sweep are not implicit extensions.

The comparison isolates the corrected package from a parent-only change. If the control cannot be reconstructed with valid provenance, record that limitation and finalize a revised bounded comparison before training; do not invent an equivalent historical baseline.

### 5. Select and confirm on development data

Measure v0.1 on the same development/regression panels before setting tolerances. Start with the previous per-language routing tolerance of 3 percentage points as a proposed engineering guard, not an immutable user requirement. Freeze the final guard before pilot training.

Select among retention-eligible checkpoints using language/source-macro factual accuracy, with explicit checks on English actor binding, Chinese unknown and unseen candidate wording. Break ties by factual NLL. Freeze slice floors and material-improvement criteria after the baseline audit and before fitting; never choose them from final results. A high aggregate cannot compensate for a near-zero essential slice.

If the candidate demonstrates material development gains and retention, confirm it with seed 2027 using the same protocol. Report both seeds; do not publish only the better one. If gains disappear or key failures persist, stop this round and retain the existing release. Do not extend an unsuccessful trajectory indefinitely.

### 6. Calibrate, evaluate and deliver

Freeze selected checkpoints and any confirmation outcome, then fit temperatures on separate calibration material. Evaluate both arms and unchanged v0.1 on the same fresh acceptance panel once. Calibration improves probability interpretation; positive temperature cannot repair wrong argmax judgments.

Report individual and complete-group correctness by language/source/domain/phenomenon, unknown confusion, unseen expressions, candidate robustness, routing retention, NLL/Brier/ECE, exposure and uncertainty. Natural panels stay separate from generated panels; explain task-specific baselines where relevant.

If results support a useful preview, package one selected shared model with explicit capabilities and limits, calibrated inference, numeric/string IDs and reproducible load examples. An eligible preview need not reach universal 80% or 95%; limitations must remain visible. If gains are only template-specific or essential slices remain unusable, preserve v0.1 and record a completed unsuccessful iteration rather than label it a general semantic release.

## Required implementation and deliverables when execution is authorized

| Area | Planned work |
| --- | --- |
| Data | Corrected world/group generator, reviewed variants, historical-component audit and frozen split manifests |
| Training | Explicit exposure schedule, common-parent control/candidate orchestration and reproducible resume metadata |
| Evaluation | Same-state query groups, unknown/wording cells, source macros, regression versus fresh-panel separation |
| Runtime | CPU/CUDA parity, candidate ordering/chunking, stable repeated-inference and validation memory, standalone latency measurement |
| Verification | Meaningful tests for gold/group integrity, split isolation, sampler coverage, resume/fallback and acceptance-opening guards; existing production and learning checks |
| Reports | Raw training loss plus labelled smoothing, fixed-probe/held-out loss, group accuracy, pre-clip gradient norms and actual exposure |
| Delivery | Selected preview only if supported; otherwise explicit no-promotion result; reproduction commands and handover |

Keep runtime artifacts under a new ignored `runs/decision/factual-v03/<run>/` root; downloads/data/weights remain ignored. Record parent/config/tokenizer/data hashes, seeds, optimizer state, traces, predictions, calibration, protocol and report. Reuse the factual benchmark machinery where sound; use new versioned data metadata instead of adding compatibility work solely for historical recipes. Historical v0.2 files remain preserved for review.

On completion, write `docs/decision/v03.md`, update README/handover/retrospective and track only a reviewed representative chart. Save successful and failed learning conclusions without inflating them into general laws. Do not create a v0.3 release directory or advertise a working command until its implementation exists.

## Stop conditions and alternative paths

- **Cannot fit:** debug targets, masks, gradients and optimization before more data or parameters.
- **Fits but new compositions fail:** revisit supervision correlations and question/actor dependence; do not assume capacity is the cause.
- **Controlled behavior improves but natural/wording transfer does not:** stop the bounded pilot and compare foundational representations in a separately authorized next plan.
- **Routing retention fails:** examine initialization and replay exposure on development data; no test-driven retuning.
- **Gains are not repeatable:** preserve both seeds and decline promotion.

The remaining possible foundation comparison is **Path C**, a public multilingual pretrained encoder with candidate scoring. It prioritizes practical semantic coverage but changes the own-weights learning objective; implementation requires a separately authorized plan. Larger from-scratch pretraining/size is a later resource-intensive option, not the default response to current failures. RL remains for later policy/multistep learning, not a remedy for missing factual semantics.

The user removed **Path B (Qwen semantic preprocessing)** from follow-up planning. It is not a fallback or an implementation task for this roadmap. Historical distillation records remain learning material, not authorization to resume that direction.

Behavioral evaluation follows [CheckList](https://aclanthology.org/2020.acl-main.442/) and [Contrast Sets](https://aclanthology.org/2020.findings-emnlp.117/). These motivate testing beyond averages; they do not establish that this proposed training package will succeed.
