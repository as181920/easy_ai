# Reusable distillation

`EasyAI::Distillation` provides offline teacher collection, resumable artifacts and candidate-level soft-target loss. Decision is the first task adapter. The collector itself has no knowledge of `state`, questions, candidate IDs or a particular model family.

The first [local Qwen screening](teacher-v1.md) is complete: 16/16 binding accuracy, 75% public-source macro accuracy, but only 91.67% candidate-order agreement against a 95% gate. It failed screening; its development labels were not used for student training.

The follow-up [task-specific prompt comparison](teacher-task-v1.md) also failed: agreement fell to 85.42%, and correctness in both candidate orders fell from 39/48 to 35/48. Both profiles remain experimental; no bulk labels or new student checkpoint are promoted.

The [from-scratch gold-supervised coverage round](../decision/coverage.md) is complete: longer training gives a limited accuracy gain with worse raw probability metrics, while wording expansion fails its gate. The next planned direction is task-aligned gold supervision and stronger conditioning/probability evaluation. Further teacher experiments are deferred; this infrastructure and its failed experiment records are retained for possible future reuse.

## What is being distilled?

Teacher logits are **not required**. This implementation supports three distinct paths:

| Teacher output | Student supervision | Implemented path |
| --- | --- | --- |
| An answer selecting a candidate | Hard pseudo-label, trained with cross-entropy | Local HTTP teacher → Decision adapter → export ordinary training rows |
| An answer on an already labeled input | Additional teacher cross-entropy while retaining gold supervision | Decision Trainer `--teacher-artifact` |
| A complete, externally scored candidate distribution | Candidate-level KL plus gold cross-entropy | Ruby teacher protocol and Decision Trainer; no built-in Qwen probability scorer yet |

A teacher-generated explanation is text, not automatically an extra training signal for a candidate classifier. It would need a separate objective, such as evidence selection, or conversion into validated new training examples. For a generative student, sequence-level distillation can use teacher-generated answers without matching teacher logits, while student training still typically uses token-level cross-entropy. See [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947).

The student still tokenizes its **inputs**. What is unnecessary here is aligning its vocabulary or hidden layers with Qwen. A teacher writing “90% confidence” is not supplying a measured candidate probability.

## Implemented boundaries

```text
Local Qwen / other chat model
            |
   Teachers::LocalHttp             model/revision/parameters
            |
        Collector  <------------- task adapter request + parser
            |
   immutable Artifact             input/teacher fingerprints; raw replies
            |
       Decision adapter
          /     \
  unlabeled     gold-labeled input
     |                |
  hard-label       gold CE + teacher CE/KL
  export              |
     +------> existing Decision Trainer
                      |
               student-only inference
```

The shared layer lives in `lib/easy_ai/distillation/`; candidate mapping/export/training integration lives in `lib/easy_ai/decision/distillation_*.rb`. Teachers implement `signature` and `call(request)`. Task adapters implement `signature`, `identity(example)`, `request(example)` and `parse(example, reply)`. A non-Decision adapter is exercised by the collector tests; generative and embedding student trainers are not implemented.

The local HTTP backend accepts only a loopback URL, uses Faraday without a proxy, and returns generated content. It retries connection/timeouts and HTTP 429/5xx up to three attempts; malformed or truncated answers fail explicitly. It uses the chat-completions protocol supported by [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server), not a Python runtime.

The Decision prompt never contains the gold target. The teacher selects a zero-based index in schema-constrained JSON; the adapter validates it and maps it to the original string/numeric candidate ID. Neither free text matching nor inferred confidence is used.

## Local teacher

The checked-in configuration is `config/distillation/qwen3_4b.yml`: [official Qwen3-4B Q4_K_M](https://huggingface.co/Qwen/Qwen3-4B-GGUF), repository revision `bc640142c66e1fdd12af0bd68f40445458f3869b`, GGUF SHA256 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`. It requests non-thinking, greedy, schema-constrained answers with at most 64 generated tokens. This is a fixed labeling protocol, not a claim that these parameters maximize teacher quality.

Runtime source is pinned to llama.cpp `b11146` / `7fe450e19305b828c199d602c23a8337aaa1f03b`. The prebuilt CUDA binary requires newer glibc than this Ubuntu installation; build from source against the installed CUDA 12.9 toolkit instead of upgrading system libraries. For an already extracted source tree:

```bash
cmake -S data/distillation/runtime/source -B data/distillation/runtime/build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_BUILD_NUMBER=11146 -DLLAMA_BUILD_COMMIT=7fe450e19305b828c199d602c23a8337aaa1f03b
cmake --build data/distillation/runtime/build --target llama-server -j 6
```

After verifying the model checksum, start a local service:

```bash
data/distillation/runtime/build/bin/llama-server \
  --model data/distillation/downloads/Qwen3-4B-Q4_K_M.gguf \
  --alias qwen3-4b-q4km --host 127.0.0.1 --port 18081 \
  --ctx-size 4096 --parallel 1 --n-gpu-layers 99 --batch-size 256 --ubatch-size 128 --jinja
```

Teacher and student should run in separate stages on the 6 GiB GPU. Stop the teacher process after collection, then train the student. If teacher placement exceeds available memory, reduce offloaded layers or use CPU; student resource recovery is unchanged. Source archives, builds and weights stay under ignored `data/`, not system directories. Exact runtime measurements and teacher audit results must be recorded separately from these commands.

## Audit before producing training labels

```bash
bundle exec ruby benchmarks/distillation/teacher_audit.rb \
  --teacher-config config/distillation/qwen3_4b.yml \
  --output runs/distillation/qwen3-4b-audit-v1
```

`--prepare-only` freezes the audit inputs without calling a teacher. Selection uses existing **validation** data: four distinct source groups per public source/label, plus two complete mixed-truth binding groups per language; every example is also queried with reversed candidates. It never reads test. This is a small, balanced development diagnostic, not a population accuracy estimate or new blind challenge set.

Before teacher execution, the script fixes gates of relation accuracy ≥95%, public-source macro accuracy ≥70%, and candidate-permutation agreement ≥95%. All metrics and predictions are retained; a failed gate is a result, not permission to quietly lower the threshold. Passing is only a screening result and does not establish that distillation improves a student.

Audit and collection accept `--prompt-profile generic` (default control) or `--prompt-profile task_specific` (explicit OCNLI, DuReader-YesNo, BoolQ and relations definitions; other sources rejected). Profiles are recorded in adapter signatures and cannot share one output directory. Gold labels are never sent. Keep the profile fixed when resuming; export and supervised training infer it from the artifact.

For a prompt-only comparison, run `benchmarks/distillation/teacher_comparison.rb --baseline BASELINE_DIR --experiment EXPERIMENT_DIR --output REPORT.json`. It verifies matching data, teacher and protocol, then derives original/reversed accuracy, agreement and both-correct rates from artifacts. See the [recorded comparison](teacher-task-v1.md) for complete commands and limitations.

## Collect, export and train

Input JSONL uses the Decision format. `target` may be absent during collection. Keep validation/calibration/test in separate source groups and supply them through `--exclude` when collecting training signals:

```bash
bundle exec ruby bin/easy-ai distill-collect \
  --data data/decision/my-task/unlabeled-train.jsonl \
  --teacher-config config/distillation/qwen3_4b.yml \
  --purpose train --output data/distillation/my-task/teacher \
  --exclude data/decision/my-task/validation.jsonl,data/decision/my-task/test.jsonl

bundle exec ruby bin/easy-ai distill-export \
  --data data/decision/my-task/unlabeled-train.jsonl \
  --artifact data/distillation/my-task/teacher \
  --output data/distillation/my-task/pseudo-labels

bundle exec ruby bin/easy-ai train \
  --config config/decision/semantic.yml \
  --tokenizer data/decision/my-task/tokenizer.json \
  --data data/distillation/my-task/pseudo-labels/train.jsonl \
  --validation data/decision/my-task/validation.jsonl \
  --output runs/decision/my-task-pseudo-labels --progress
```

These are path templates: first prepare the task data and its train-only tokenizer. Pseudo-label export accepts **unlabeled inputs and hard teacher labels only**, preserves source groups and adds label provenance; it rejects overwriting gold labels or silently collapsing a soft distribution to argmax. Exported rows use the ordinary candidate CE training path. Validation should retain independent trusted labels.

For already gold-labeled training inputs, collect a separate matching artifact and use:

```bash
bundle exec ruby bin/easy-ai train \
  --config config/decision/semantic.yml \
  --tokenizer data/decision/semantic-public/tokenizer.json \
  --data data/decision/semantic-public/train.jsonl \
  --validation data/decision/semantic-public/validation.jsonl \
  --teacher-artifact data/distillation/semantic-public/teacher \
  --teacher-weight 0.5 --output runs/decision/semantic-teacher --progress
```

Hard targets use `gold CE + weight * teacher CE` with temperature 1. Soft targets use `gold CE + weight * T² * KL(teacher_T || student_T)` and allow `--teacher-temperature`. Training loss therefore differs from gold-only validation loss. If hard teacher labels equal gold everywhere, the added term merely rescales the same objective; it does **not** transfer new knowledge. A useful subsequent experiment needs additional independently varied inputs or informative soft targets.

The current HTTP adapter produces hard labels only. A custom Ruby teacher can return `kind: candidate_probabilities`, a complete string-ID probability mapping, `temperature: 1`, and a nonempty `scoring_protocol`; its signature must identify the actual scorer. Values must be finite, nonnegative, complete and normalized. The loss re-tempers in log space, preserves zeros and excludes padded candidates. It does not need teacher token IDs.

Resume requires the same teacher artifact, weight and temperature as the original run; pass the flags again. Removing or changing them fails before training. Validation/model selection remains gold-only, dynamic candidate resampling is rejected, and best checkpoints remain usable for ordinary student inference without a teacher.

## Artifacts and recovery

`contract.json` fixes source fingerprint, task adapter, teacher identity/configuration and purpose (`train` or `development`). Each successful reply is atomically cached; an interrupted collection resumes without requerying completed examples. A lock prevents simultaneous writers to one output. Completed `records.jsonl` has a SHA256 in `manifest.json`; loading checks it and indexes file offsets rather than loading every response into memory. Development artifacts cannot feed training or pseudo-label export.

Failures write `failure.json` and leave the collection incomplete. A different input/configuration needs a new output directory. Candidate identity includes texts as well as IDs, and cached requests include ordering. Gold labels are not included in teacher requests, but source dataset fingerprints protect training/restore provenance. Declaring a purpose does not discover semantic duplicates automatically: upstream source grouping remains necessary.

Tests cover scalar KL agreement, padding gradients, option reordering, malformed distributions, content parsing, HTTP failure handling, interrupted collection, artifact corruption, gold preservation, unrelated task adapters, and student resume. Run `bundle exec ruby -Itest test/easy_ai/distillation/distillation_test.rb` or the complete `bundle exec rake test`.

Background: [original design](../decision/distillation.md) and [training retrospective](../decision/retrospective.md). The existing from-scratch experiments remain baselines; infrastructure tests do not establish semantic improvement from a teacher.
