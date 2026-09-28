# Task-specific teacher prompt: controlled comparison

Date: 2026-09-28. Follow-up to the [generic Qwen teacher screening](teacher-v1.md).

**Result: explicit task definitions did not solve label instability. This profile failed screening and regressed on the fixed development examples. It is retained as an experiment, not promoted for bulk labeling. No student weights changed.**

## Hypothesis and control

The generic instruction might leave the teacher uncertain about the difference between NLI, answer-stance classification and passage-based yes/no questions. The experiment appended a source-specific task definition to the original system prompt:

- OCNLI: distinguish supported, contradicted and undetermined hypotheses; unmentioned is not automatically false.
- DuReader-YesNo: classify the supplied answer's stance rather than answer the question independently; distinguish affirmative, negative and conditional answers.
- BoolQ: answer using the passage, resolving referents, negation and qualifications.
- Relations: bind the queried fact to the named person and account for assertion polarity.

These are general instructions, with no gold targets or worked examples from the audit. They are written in English, like the control prompt. The source selects a task definition; it is not given to the student as an extra feature. This experiment tests the specific definitions in `DistillationAdapter::TASK_INSTRUCTIONS`, not all possible task-specific prompts.

Everything else stayed fixed: Qwen3-4B Q4_K_M weights, llama.cpp build, non-thinking greedy decoding, output schema, seed, context, GPU placement, input order and the predeclared gates. Model, server and server-library checksums were reverified. The 48 originals and their 48 candidate reversals have the same input SHA256 as the control:

`63fcc527831d30b3374153d01edaa885ae9c3fbb848e2c487815bc93b7f36c9d`

This is reused development data, already inspected after the first run. It is neither a blind benchmark nor evidence of population-level differences. Each pair shares an input; 96 requests are not 96 independent examples.

## Results

| Metric | Generic control | Task-specific | Original gate |
| --- | ---: | ---: | --- |
| BoolQ original accuracy | 6/8 (75%) | 6/8 (75%) | Public macro component |
| DuReader original accuracy | 9/12 (75%) | 8/12 (66.67%) | Public macro component |
| OCNLI original accuracy | 9/12 (75%) | 9/12 (75%) | Public macro component |
| Public-source macro accuracy | 75% | 72.22% | ≥70%: both pass |
| Relations original accuracy | 16/16 (100%) | 16/16 (100%) | ≥95%: both pass |
| Candidate-order agreement | 44/48 (91.67%) | 41/48 (85.42%) | ≥95%: both **fail** |
| Correct in both orders | 39/48 (81.25%) | 35/48 (72.92%) | Additional diagnostic |
| Original / reversed total correct | 40/48 / 39/48 | 39/48 / 38/48 | Additional diagnostic |

Fourteen pairs changed at least one answer, all Chinese in this selection. Across the 96 individual requests, eight wrong answers became correct and ten correct answers became wrong. Chinese agreement declined from 28/32 to 25/32; Chinese correctness in both orders declined from 25/32 to 21/32. English predictions were unchanged on these 16 pairs.

The more explicit NLI instruction fixed the unsupported “popular singers” inference in both orders. The conditional work-experience answer also became correct in both orders. However, other previously correct NLI and DuReader answers changed to unknown/conditional labels. One Chinese relation example changed from correct in both orders to correct only in the original order. These observations do not establish whether every disagreement comes from teacher reasoning, annotation ambiguity or the wording of these instructions.

The task-specific profile agreed with itself on six pairs where both answers were wrong. **Agreement is a robustness diagnostic, not proof of a valid training label.** A filter that keeps agreement alone would still admit known errors. Original-order aggregate accuracy also hides the new reversed-order relation error.

## Implementation and resources

`--prompt-profile generic|task_specific` is available to the audit and `distill-collect`. Generic stays the default control; neither profile has passed screening. Task-specific collection rejects unknown sources instead of silently using another task. The adapter signature stores the complete task definitions, and row identity records the task source. Changed profiles cannot reuse the same collection; export and teacher-supervised training reconstruct the matching adapter from its signature.

The new `benchmarks/distillation/teacher_comparison.rb` derives metrics from the original datasets and checked teacher artifacts, without relying on previously printed summary values. It rejects different teacher configurations, input fingerprints or audit protocols, verifies that reversal only changes candidate order, and reports per-source/per-language accuracy, agreement, both-correct rates and individual transitions. Its scope is a prompt-only comparison.

The local teacher used 3072 MiB after loading and 3082 MiB after the audit (samples, not peak measurements). The collection interval between contract and manifest file timestamps was about 13.01 seconds, excluding model loading. Reported API usage was 21,926 prompt tokens and 576 completion tokens, compared with 14,782 and 576 for the control. Prompt totals include cache reuse and are not unique-compute measurements. The service was stopped afterward; cache replay succeeded with it stopped. No installation, model download, CPU fallback or student training was needed in this comparison.

Verification: **101 tests / 1268 assertions**, learning **7 tests / 14 assertions**, RuboCop **111 files**, all pass. New coverage checks profile/gold separation, unknown task rejection, artifact profile/source alignment, profile-aware export/training, comparison controls and stable-but-wrong predictions. Cache replay and comparison require no running teacher. Original learning plots remain intact.

## Learning and next experiment

1. Clearer human instructions are a testable hypothesis, not a guaranteed model improvement. Keep corrected cases and regressions together; inspecting only the original failures would have made this prompt look more successful than it was.
2. Longer instructions increased reported prompt tokens without improving these metrics. Do not attribute this outcome to insufficient student capacity: no student ran here.
3. Keep gold labels intact and retain both failed teacher configurations. Do not lower the gate, silently select the better prompt per example, or treat stable pseudo-labels as gold.
4. Stop this prompt comparison here. A useful next bounded teacher experiment is whether allowing a fixed reasoning budget improves the **generic control** on the same diagnostic. Keep the weights and task prompt fixed, separately record latency/output tokens and truncations, and never count an incomplete response as a label. This would compare inference protocols, so the current prompt-only comparison command intentionally cannot compare those different teacher signatures. It has **not been run**.
5. If a configuration passes development screening, freeze it before evaluating additional unused development groups. An independently isolated student challenge set is still outstanding. Bulk distillation and claims of student improvement must wait for usable supervision and a matched student comparison.
6. Public gold-supervised learning remains available independently. For extra teacher-generated coverage, use training-source paraphrases with polarity/subject/modality checks and review. New valid inputs can add information; repeating a teacher answer that already equals the gold label only repeats the existing CE target. Paraphrase generation and its validation pipeline remain unimplemented.

## Reproduce and hand over

Start the pinned local teacher as described in the [implementation guide](README.md), then run:

```bash
bundle exec ruby benchmarks/distillation/teacher_audit.rb \
  --teacher-config config/distillation/qwen3_4b.yml \
  --prompt-profile task_specific \
  --output runs/distillation/qwen3-4b-audit-task-v1

bundle exec ruby benchmarks/distillation/teacher_comparison.rb \
  --baseline runs/distillation/qwen3-4b-audit-v1 \
  --experiment runs/distillation/qwen3-4b-audit-task-v1 \
  --output runs/distillation/qwen3-4b-audit-task-v1/comparison.json
```

Existing complete artifacts replay without a teacher. Use a new directory for fresh inference; `--prepare-only` freezes data and gates without making requests. The task-specific artifact fingerprint is `a0d478c91fc23e2d62e8ec5330057e90857a17b2c3156792aa2182e1e4ae012a`.

Local evidence is under `runs/distillation/qwen3-4b-audit-task-v1/`: data, protocol, raw teacher records/cache, summary, comparison, runtime measurements and server/audit/replay logs. Both experiment directories and the pinned runtime/weights are ignored by Git; copy them separately for handover. Git contains code and the learning record. Existing Decision checkpoint paths and earlier learning charts remain unchanged.
