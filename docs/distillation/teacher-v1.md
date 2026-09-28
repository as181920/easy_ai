# Qwen teacher screening: first local run

Date: 2026-09-28. This is a development screening result for a particular teacher, prompt and quantization. It is not a student distillation result or a general Qwen benchmark.

Follow-up: the [task-specific prompt experiment](teacher-task-v1.md) has now been run and also failed screening. This page preserves the original generic-prompt result and its then-proposed next steps.

**Result: the teacher failed the predeclared candidate-order stability gate. No student was trained on these audit labels, and no new Decision checkpoint is promoted.** The reusable collection/export/training implementation is separately verified by tests.

## Setup fixed before inference

- Teacher: official `Qwen/Qwen3-4B-GGUF`, Q4_K_M; revision `bc640142c66e1fdd12af0bd68f40445458f3869b`.
- Model SHA256: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` (verified after download).
- Backend: llama.cpp `b11146`, source commit `7fe450e19305b828c199d602c23a8337aaa1f03b`; locally compiled with CUDA 12.9, compute capability 7.5.
- Decoding: non-thinking, greedy, maximum 64 output tokens, schema-constrained integer answer. The mapping from answer index to candidate ID happens in Ruby.
- Context: 4096 tokens, one server slot, GPU layers requested 99, batch 256 / microbatch 128. Loopback endpoint only; no system installation.
- Data: existing public and relation **validation** splits, never test. Fixed before teacher inference: four distinct groups per public source/label and two complete mixed-truth binding groups per language.
- 48 original examples plus the same 48 with reversed candidates. Gold labels are retained for evaluation and are not sent in teacher requests.

The selection contains 8 BoolQ, 12 DuReader-YesNo, 12 OCNLI and 16 relation examples. Balanced label selection makes these diagnostic samples different from the population distribution. Reversed examples are dependent checks, not another 48 independent samples.

Audit input SHA256: `63fcc527831d30b3374153d01edaa885ae9c3fbb848e2c487815bc93b7f36c9d`.

## Results

| Metric | Observation | Predeclared gate |
| --- | ---: | ---: |
| BoolQ, original order | 6/8 = 75% | Included in public-source macro |
| DuReader-YesNo, original order | 9/12 = 75% | Included in public-source macro |
| OCNLI, original order | 9/12 = 75% | Included in public-source macro |
| Public-source macro accuracy | 75% | ≥70%: pass |
| Subject binding, original order | 16/16 = 100% | ≥95%: pass |
| Same prediction after reversing options | 44/48 = 91.67% | ≥95%: **fail** |
| Overall screening | — | **Failed** |

Original-order accuracy across all tasks is 40/48; reversed-order accuracy is 39/48. The small difference in aggregate accuracy hides four changed predictions, including cases where both answers are wrong. Chinese order agreement is 28/32 = 87.5%; English is 16/16 = 100%. This does not establish reliable English performance beyond these examples.

All four inconsistencies come from Chinese public tasks: two DuReader and two OCNLI. They involve conditional answers, unsupported claims, or pragmatic interpretation. One OCNLI state describes fashionable clothing, makeup and hairstyles; the hypothesis asserts that the people are popular singers. The gold target is insufficient information, but the teacher switches from entailment to contradiction under candidate reversal. The unmodified inputs, labels, responses and candidate mappings remain in the local audit artifacts.

These observations support caution about the **labeling setup**, not a universal claim that 4B models or content distillation cannot work. Prompt interpretation, quantization, model limitations and source annotation ambiguity have not been separately isolated. Candidate reordering should not change the task's gold answer regardless of these possible causes.

## Resource and implementation evidence

The teacher ran on the Quadro RTX 3000 6 GiB GPU. Process memory was 3072 MiB after loading and 3084 MiB after the audit; these are samples, not measured peaks. The service was stopped after screening to free resources. API usage reported 14,782 prompt tokens and 576 completion tokens across 96 requests; prompt counts include cache reuse and must not be interpreted as unique compute.

The downloaded prebuilt CUDA runtime required glibc 2.38, while this system has 2.35. Building the pinned source against the existing CUDA 12.9 toolkit solved runtime compatibility without changing system libraries. The first model transport stalled; parallel resumable transfer completed and the final file matched the official checksum. All downloads, source/build files, artifacts and logs are ignored by Git; `.gguf` is also explicitly ignored.

Code verification: **95 tests / 1229 assertions**, learning **7 tests / 14 assertions**, RuboCop **109 files**, all pass. Tests include a complete content → pseudo-label export → student train → checkpoint reload workflow using a fixture teacher. This proves wiring and recovery behavior, not real-teacher semantic improvement.

## What this teaches us

1. **Content-only distillation is technically sufficient for this task.** A teacher's selected option becomes a hard target for candidate CE; teacher token logits are unnecessary. This run measured the resulting labels, not hidden representations.
2. **Teacher output is not ground truth.** A larger pretrained teacher can pass a small binding task and still produce unstable natural-language labels. Blindly replacing trusted labels would introduce known errors.
3. **Aggregate accuracy misses behavior changes.** Original/reversed totals differ by only one correct item, but four examples change meaning. Keep permutation checks and per-language metrics.
4. **Repeating known gold labels supplies no new knowledge.** Where teacher and gold agree, adding hard teacher CE only rescales the same objective. New semantics need more varied valid inputs, informative soft targets, or another explicit objective.
5. **The route is not rejected by one failed teacher configuration.** The result rejects automatic acceptance of this configuration under the fixed screening protocol. It does not justify changing the gate after looking at results.

## Next experiment options

Keep the from-scratch model as a baseline. Before collecting a large training corpus, compare a clearly task-specific labeling prompt with this generic prompt on development data, keeping this failed run intact. Define NLI entailment/contradiction/unknown separately from DuReader answer stance; do not collapse uncertain or conditional cases into No.

For new training coverage, Qwen may generate paraphrases of **training-source** examples whose labels are independently verified. Preserve source-family grouping, reject changes to polarity/subject/modality, and audit accepted examples. Agreement under candidate permutation can filter unstable labels, but agreement alone does not prove correctness. Report acceptance by language/source/label so filtering does not silently remove the difficult unknown/conditional cases.

If comparing another teacher size, prompt or thinking mode, use separate artifacts and record additional compute. Do not silently change several factors or mix label-generation protocols. A fresh isolated challenge set is still needed for subsequent student generalization claims; the present screening is development data.

Once teacher/data quality is sufficient, compare the same student and update budget on gold-only data versus added validated coverage. Compare soft-target supervision separately if a genuine candidate scorer is implemented. No bulk pseudo-label collection, prompt revision, larger-model download, or student quality comparison was performed in this run.

## Reproduce and hand over

Start the local teacher using the [implementation guide](README.md), then:

```bash
bundle exec ruby benchmarks/distillation/teacher_audit.rb \
  --teacher-config config/distillation/qwen3_4b.yml \
  --output runs/distillation/qwen3-4b-audit-v1
```

The existing output resumes/reuses the cached responses and checks input/configuration fingerprints. To measure a fresh inference run, choose a new output directory. Use `--prepare-only` to generate and freeze the input selection without inference.

Local evidence: `runs/distillation/qwen3-4b-audit-v1/{protocol.json,development.jsonl,runtime.json,summary.json,audit.log,teacher/}` and `runs/distillation/qwen3-4b-server.log`. Teacher artifact fingerprint: `256d8cddfcff62a89d3995c2ef6644d3357f50313bd062f1f48417961d038fc2`. Runtime metadata includes source-archive and compiled-server hashes. Copy ignored artifacts separately when handing over; Git contains the implementation and this result summary, not model weights.
