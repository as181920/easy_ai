# Decision v0.4 plan — reviewed supervision and factual learning

Status: **completed bounded implementation, 2026-10-03**. The user adopted this plan as a development goal. See [v04.md](v04.md) for the actual audit, implementation, fitting failure, runtime verification and deviations. The training-only actor-binding check failed at its 1,000-update cap, so pilots and acceptance remain unrun under the stop rule below. v0.3 is committed as `af869a1`; v0.1 remains delivered. Read [v03.md](v03.md) and [retrospective.md](retrospective.md) for prior evidence.

## Recommendation and intended progress

Audit supervision, establish known-judgment and actor-binding learning, then add unknown judgments without losing those skills. Keep our own initialization and Ruby/Torch.rb. Do not begin with more corpus rows, a larger model, another calibration sweep or external weights.

The intended product is **one shared Chinese/English checkpoint for judging a claim against a short supplied record**, returning candidate probabilities. Initially scope records to roughly one to four explicit propositions and single-step support/contradiction/missing-evidence judgments across multiple domains. No language argument, numeric/string candidate IDs and Ruby-assembled JSON remain supported. This is not a general question-answering model, future-event forecast or business integration.

Publish a useful scoped preview only if independent measurements support it. Keep the 80% direction advisory; improvement must appear in known judgments, unknown handling and complete contrast groups, not just a source macro. A prospectively defined binary-only preview is an acceptable intermediate deliverable if known judgments become reliable but three-state scoring does not. It cannot detect missing information or automatically determine whether a request is answerable.

## Evidence determining the next intervention

v0.3's fresh factual macro rises to 52.23%, while factual row accuracy falls to 31.99%. Its 228-row isolated fit reaches 100%, but the same training-example probe during the mixed pilot finishes at only 60.96% overall, factual NLL 0.7693 and **0% mixed-truth binding in both languages**. Therefore, failure is not confined to unseen evaluation expressions. Isolated fitting proves a functioning optimization path, not mastery of the larger mixture.

A post-v0.3 diagnostic uses saved logits on **gold-known cases** only:

| Candidate v0.3 diagnostic | English | Chinese |
| --- | ---: | ---: |
| Known three-option cases | 3,200 | 3,200 |
| Original three-option accuracy | 5.59% | 7.53% |
| Unknown chosen on those known cases | 89.56% | 88.19% |
| Supported/contradicted ranking after excluding unknown | 52.09% | 44.22% |
| Existing two-option known-case accuracy, 800 rows/language | 54.75% | 54.25% |
| Mixed-truth binding with binary ranking | 15.87% | 6.25% |

Method: for a known row, select the maximum saved logit among option IDs `yes` and `no`, then score both members of each mixed actor-switch pair. This is a retrospective diagnostic, not an inference repair: filtering to known rows uses evaluation gold, and no deployment system knows those labels. Positive temperature cannot change these rankings. The candidate prediction file SHA is `da3e6754831487b9769a3734484cfb14315efb320f1514b30f58c09ebf6885fa` under `runs/decision/factual-v03/pilot/`.

These observations support testing basic judgment learning and supervision quality before treating unknown calibration or model capacity as the main cause. They do not prove that any particular dataset defect explains the failure.

```text
v0.3 committed; predictions now observed regression
                         |
           Independent supervision audit
                         |
      Reviewed labels, scope and counterfactual families
                         |
       Freeze canonical options + fresh evaluation
                         |
          Common own-v0.1 parent / unchanged model
             |                            |
      Current supervision         Reviewed supervision
             +------------+---------------+
                          |
      Stage 1: known judgments -> actor/event binding
                          |
      Stage 2: add unknown, preserve known judgments
                          |
      Development selection + conditional confirmation
                          |
          Calibration / one-time fresh acceptance
                          |
       Three-state preview / binary preview / no promotion
```

## 1. Audit real supervision before training

Export about **200 target-blind reviewed examples**, stratified across language, public/generated source, target, question polarity and candidate count, plus complete contrast families. Review the record, question and every option before revealing the stored target. Check full source context where needed. Preserve original labels, proposed labels, reviewer decisions, ambiguity/exclusion reasons and examples of each defect; report rates and denominators by source rather than only a global pass rate. This sample is an initial screen, not a guarantee about the whole corpus. Repeated defects require auditing their complete construction/adapter family.

Audit these separately:

- **Gold correctness and answerability:** conflicting labels for identical inputs, implicit assumptions, missing context, truncation, negation scope, pronoun/actor referents and whether a negative judgment has actual contrary evidence.
- **Source semantics:** OCNLI neutral, absence of a recorded fact and DuReader conditional “depends” are not interchangeable concepts. Review each adapter's rendered question/options. Do not force conditional/modality records into the initial three-state claim profile; exclude them from that profile or keep their original task semantics in separately reported supervision.
- **Candidate meaning:** the answer describes support for the queried claim, including negative claims; it must not silently switch to the event's positive occurrence. Review Chinese/English wording and translations. Start with fixed reviewed candidate descriptions, so unseen option semantics cannot obscure whether core evidence matching was learned.
- **Construction diversity and shortcuts:** count independent proposition/context families, not generated names. Check joint actor/event/order/polarity/label distributions, duplicates and lexical identity associations. Audit both flips and invariances using declared worlds and independent checks, not keyword counts as semantic evidence.
- **Metadata accuracy:** v0.3 fact-flip rows change only the first actor, and their `truth_pattern` retains initial family values although rendered facts can change. Distinguish base-family metadata from actual-row facts in future slice reports. Labels and primary mixed actor-switch metrics are not thereby shown wrong; this is a coverage/reporting issue to resolve.

Native-language/source review should adjudicate ambiguous cases; unresolved examples stay out of supervised training. Do not claim that the model itself or a teacher independently verified its own labels. Use public caches already available where suitable; audit source/licensing/history before adding any new corpus. No download or teacher run is implied by this plan.

**Exit:** a documented label contract, reviewed manifest, consistent source adapters, correction/exclusion list and semantic-family coverage report. Resolve systematic errors before pilots. If manual/source review is not available for a critical ambiguity, record the missing input rather than guessing gold.

## 2. Build a compact corpus that teaches relations

Use an explicit record/claim schema with support, contradiction and insufficient-evidence targets. Metadata is for gold derivation, sampling and evaluation only; model inputs remain state, question and candidate text.

For each controlled actor/event frame, reuse lexical identities across **all four two-actor truth assignments**, independently flip either actor's fact, change the queried actor, reverse order and negate the assertion. Include same-truth and mixed-truth records. This prevents an actor's name or one fixed world assignment from standing in for evidence. Add same-actor/different-event cases: buying a ticket and booking a room must not collapse into one predicate.

Unknown families must include both an absent actor and **a present actor whose queried event is unrecorded**. Within a record, provide supported, contradicted and unsupported claims; alternate positive/negative assertions and vary distractors independently. A false recorded event is negative evidence, not missing evidence. Leave conflicting/time-dependent records, modality and multi-step inference outside the initial scope.

Start with roughly **1k–4k controlled decisions per language**, subject to whole-family accounting, plus reviewed natural records and routing replay. The point is enough repetition to learn balanced relations within the available update budget, not that smaller data is inherently better. Count frames, rendered variants, unique rows, visits and tokens separately. Expand diversity only after the first stage learns the existing relations.

Two arms share canonical question/candidate formatting, curriculum, parent, architecture and optimization:

- **Control:** audited existing supervision, with only mandatory label/schema fixes and matched eligible sampling. Preserve its construction distribution as the comparison reference; do not retain known invalid labels merely to weaken it.
- **Candidate:** reviewed relation families and qualified natural supervision that address the audit's coverage/meaning defects.

Match language/target/profile allocation and size where practical. Record exclusions and unmatched exposures explicitly. This tests a reviewed supervision/coverage package under a common learning recipe; it does not isolate every change or reproduce v0.3's flat-mixture run. The unchanged v0.3 weights remain a regression reference.

## 3. Freeze evaluation and the label contract

Before pilots, reserve complete counterfactual frames, paraphrases, translations and public source components. Same record/claim material must not cross splits. Reusing ordinary lexical items is allowed; evaluate new actor/event combinations independently from new wording or vocabulary. New names alone are insufficient. All previously inspected v0.1–v0.3 panels are regression only.

Maintain separate views on unseen families:

1. Canonical descriptions and familiar construction vocabulary, testing composition/evidence dependence.
2. Reviewed held-out state/question expressions, with canonical candidates.
3. Held-out candidate descriptions, changing options while holding the record/claim fixed.
4. Natural short-record judgments within the declared scope, plus broader public QA/NLI as separately labelled transfer diagnostics.
5. Existing routing/factual regression.

Proposed independent units: at least 100 controlled frames/language for development, separate calibration families, and about 200 acceptance frames/language; aim for 200 eligible natural decisions/language and at least 200 routing groups/language. Freeze actual counts after the history audit, documenting shortfalls. Do not silently shrink the product scope after observing acceptance. Strict length limits require coverage reporting, not hiding unsupported inputs.

Define a canonical three-choice profile and a two-choice known-evidence profile before training. One bilingual checkpoint handles either without a language flag. IDs carry no semantic meaning in the neural input; randomize option order during training and test it. Unseen arbitrary candidate wording remains unvalidated until its separate view passes.

## 4. Use bounded stages with measurable mastery

Run a small verified training fit first, capped at 1,000 updates and using no evaluation examples. Include both queried actors, both fact flips, mixed/same worlds and multiple events. Check loss/gradients and parameter changes before and after validation; tiny-set fitting weights must not initialize pilots.

Pilot defaults: delivered own v0.1 weights/tokenizer, unchanged 6.63M network, ordinary candidate CE, FP32, microbatch 4 / accumulation 8, CUDA first with 4,096 MiB process budget and supported CPU fallback. Freeze LR/warm-up and any fallback adjustments before comparison. Seed 1337, **at most 2,000 updates per arm**, validation every 100. No margin, auxiliary head, dynamic growth, MLM, teacher or RL in this iteration.

- **Stage 1, at most 1,000 updates:** known judgments with two options, progressing from single facts to actor/event contrasts. Train on supported/contradicted examples only; never turn unsupported gold into a false answer because its option was removed. Balance both targets within languages. Use actual training-family mastery to advance, with independent development checks against memorization.
- **Stage 2, remaining budget:** introduce three choices and missing-evidence cases while replaying known contrasts. Balance supported, contradicted and unknown **within controlled three-option exposure**, rather than treating an entire known source as one class. Keep supported/no-unknown, contradicted/no-unknown and unknown recall visible separately.

Proposed common mixture is 60% controlled factual, 20% qualified natural supervision and 20% routing replay, exact over a deterministic update cycle. Binary-stage natural examples must be genuinely known; neutral/conditional rows cannot be relabeled to fit two choices. Freeze resulting eligibility/counts before training and log actual target/language/candidate-count visits. Complete same-state claim sets should be sampled together where practical.

Advancement proposals: balanced training-probe known accuracy at least 95%, mixed-group correctness at least 90%, and clearly above-chance independent canonical development behavior for two consecutive checks. Finalize these engineering diagnostics from the audited baseline before pilots; they are not universal semantic benchmarks. **If known learning does not pass within its budget, stop that arm before adding unknown.** If unknown improves while known/binding collapses, retain the pre-unknown checkpoint and record the joint-stage failure. Do not extend steps or sweep seeds until something passes.

## 5. Select a useful capability, not a flattering metric

Primary three-state measure: per-language **macro recall across supported, contradicted and unknown gold classes**, with the worst language reported first. Always predicting unknown scores one third on this measure. Supplement it with factual row accuracy, NLL, per-class confusion, complete mixed groups, state/query sensitivity, and domain/wording results. Do not select on the old known-source/unknown-source macro alone.

Preliminary preview floors for the declared scope: at least 70% known and unknown recall per language, at least 50% mixed-truth complete-group correctness, and no material regression on the reviewed natural judgments inside scope. Aim toward 80% balanced accuracy without treating it as an immutable universal gate. Freeze final floors and paired routing tolerance before pilots, using adequate development groups; report uncertainty rather than equating one error on a tiny panel with a large capability shift.

Define separate binary and three-state checkpoint selection in advance. Proposed binary floors are at least 75% balanced known accuracy per language and 50% mixed-truth complete groups on independent in-scope records; it must explicitly document conditional two-option probabilities and lack of unknown detection. Routing v0.1 remains available separately if a factual preview cannot retain routing; do not claim its factual checkpoint replaces the routing release. Any narrower preview contract must be defined before final evaluation.

Select the best eligible profile/arm on development, ranking the worst-language balanced measure and breaking close ties in favor of the simpler recipe and lower NLL. Either arm may be useful. Trigger seed-2027 confirmation only if it passes development floors and materially improves over the unchanged parent (proposed five percentage points on the profile's primary balanced measure, with gains in both languages). Claim a reviewed-data benefit only if its paired comparison also improves over the control; otherwise report that the common curriculum may have helped without attributing success to data review. Confirm the selected recipe/profile, report both seeds and launch no additional seed if confirmation fails.

Freeze selected profile/checkpoints before calibration and one-time fresh acceptance. Calibrate separately for two versus three candidates if the calibration data supports that policy, using profile/candidate count rather than mandatory language input. Calibration cannot repair rankings. Apply publication rules prospectively; no best-test checkpoint selection or post-test threshold rescue.

## Implementation, tests and deliverables

Reuse sound v0.3 loader, optimizer, runtime, protocol and reporting machinery. Keep frozen v0.3 artifacts; iterate forward without adding legacy compatibility. Suggested new code:

```text
lib/easy_ai/decision/data/
|-- quality_audit.rb          # declared semantics, integrity and review manifests
`-- judgment_corpus.rb        # reviewed record/claim frames and source adapters
lib/easy_ai/decision/
`-- judgment_trainer.rb       # staged CE, balanced visits, deterministic resume
benchmarks/decision/
|-- data_review.rb            # target-blind export / adjudication import
`-- judgment.rb               # freeze, fit, staged arms, select, calibrate, report
docs/decision/v04.md           # actual results, chart, loading and limitations
runs/decision/judgment-v04/    # ignored data, reviews, weights and raw traces
```

Tests must verify rendered text against explicit gold for representative reviewed constructions; either-actor/event counterfactuals; complete-group leakage; unknown versus negative evidence; candidate meanings/counts; equal controlled target exposure; per-class metric baselines; resume across stage boundaries; parameter identity/updates across validation; and acceptance guards. Include deliberately broken labels/metadata as negative fixtures so the audit demonstrably detects them. Retain effective-gradient, CPU/CUDA, permutation/chunking, JSON and memory checks. Do not use lexical rules as model predictions.

Deliver: quality audit with reviewed examples and rates; frozen protocol/source/data hashes; original versus reviewed exposure; per-stage training and held-out curves; raw/calibrated predictions and group intervals; runtime verification; and **a measured factual preview if supported**. Track one reviewed figure and documentation; keep downloads/reviews/weights ignored. Update README, handover and retrospective. A completed unsuccessful round must identify the failed stage, not merely repeat “more semantics needed.”

## Decision at each outcome

| Observation | Next action |
| --- | --- |
| Systematic label/meaning defects | Correct affected families and re-audit before training |
| Reviewed core cannot fit | Stop; diagnose tokenization, masks, live parameters, loss, gradients and bounded optimization |
| Fits training but fails canonical unseen combinations | Record relation-learning failure; propose a separate architecture/representation comparison using the reviewed corpus |
| Canonical judgments work, paraphrases/options fail | Scope the preview prospectively; extend supervised expressions with a new independent evaluation |
| Binary works, unknown destroys it | Deliver binary only if its predeclared acceptance passes; investigate three-state supervision separately |
| Reviewed supervision works across scope | Confirm, calibrate, package and provide reproducible inference examples |

The fallback is a **separately authorized** representation comparison, not an automatic second project. An own-weight joint record/claim encoder comparison or public pretrained encoder can be considered then, with audited data held fixed. Path B remains removed. Do not add business orchestration, RL, external teacher preprocessing or unrestricted capacity/compute sweeps to this plan.

## Research context

Hypothesis-only predictors can exploit annotation artifacts in NLI data, motivating input-removal references and source review: [Annotation Artifacts in Natural Language Inference Data](https://aclanthology.org/N18-2017/). High standard-test performance can also conceal lexical/syntactic shortcuts, motivating controlled relation contrasts: [Right for the Wrong Reasons / HANS](https://aclanthology.org/P19-1334/). These results support the audit design; they neither diagnose our checkpoint's exact mechanism nor guarantee this curriculum will succeed. HANS is not automatically added as multilingual training or blind acceptance data by this plan.
