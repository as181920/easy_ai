# Decision 开发交接：人物绑定与下一轮优化

## Completed goal — Decision v0.3 (2026-10-03)

The user authorized [v03-plan.md](v03-plan.md), Path A only. The bounded goal is complete; read [v03.md](v03.md) before continuing. Both ordinary-CE arms start from the exact delivered own v0.1 weights/tokenizer, keep the 6.63M architecture and finish 2,000 CUDA updates / 64,000 visits. The candidate changes factored actor/query/order/polarity supervision, reviewed expressions and explicit unknown exposure. Neither has an eligible development checkpoint; confirmation is skipped. Calibration, fresh acceptance, old regression, runtime and charts are complete. **No v0.3 model is promoted; v0.1 remains delivered.** The 80% direction remains advisory, but basic known/binding failures cannot be hidden by aggregate gains.

Fresh 11,810-row panel: factual source/language macro **41.63% parent / 42.70% control / 52.23% candidate**. Candidate natural QA/NLI is **64.50% EN / 56.75% ZH**, but known judgments only **15.43% / 16.88%**, mixed-truth binding **0.72% / 0%**, and complete 26-row worlds 0%. Factual row accuracy falls from 35.52% parent to 31.99% candidate. Macro gives known and unknown sources equal weight; the unknown gain masks many wrong known judgments. Routing regression remains roughly retained at 78.68% / 79.85%, which does not repair factual failures. Old 7,144-row v0.2 regression macro: 46.33% parent / 67.62% control / 53.96% candidate. The actual v0.2 checkpoint is not evaluated on the new panel; the control uses its recipe with the common v0.1 parent.

Artifacts: `runs/decision/factual-v03/pilot/`, including protocol/baseline snapshots, `fit/result.json`, both summaries/traces/coverage, `confirmation.json`, calibration, `acceptance-opened.json`, predictions/evaluations, `report.json`, separate runtime files and reporting-only position references. The acceptance is opened and **must not be reused as fresh evidence**. `--phase report` derives saved predictions without reopening it; `--phase replay --source ... --output ...` is explicitly regression-only. Interrupted training supports deterministic resume; completed summaries refuse silent overwrite. No further training is authorized by this completed goal.

Diagnostic candidate load: `EasyAI::Decision::Predictor.load("runs/decision/factual-v03/pilot/candidate-1337/choice", device: :auto)`; strings also work, no language argument is needed, IDs accept integers/strings. This loads temperature 1; `calibration.json` separately records evaluated T=2.05843. Candidate SHA: `9c31c418d2377205124a3558e3a965c1eba50b338e79a1418f8607cecbef7a6d`. No `v0.3-factual-preview` exists. Keep `Release.load("runs/decision/v0.1-preview", device: :auto)` as the delivered API for its documented request-domain scope.

Preserved corrections: reject the first generator's deterministic relevant/irrelevant truth correlation, interrupt its control at logged update 256/checkpoint 200, before candidate or acceptance; preserve `rejected-correlation/`. Corrected `factual-v03-r2` independently draws distractor truth and fits 228 training rows at updates 600/700; fitting weights never initialize either pilot. An actor-blind reference exposes 50% overall binding with 0% mixed binding. The added 40% mixed-truth conjunction is disclosed in `protocol-before-mixed-guard.json` and the protocol amendment during control, before candidate/final evaluation. `control-1337/selection-audit.json` confirms no formerly eligible checkpoint; no control updates are rerun. Its historical trace lacks mixed metrics; final evaluation provides corrected fields for every model.

Verification: **208 tests / 2,531 assertions**, learning **13 / 56**, RuboCop **184 files, no offenses**. Gold/group integrity, independent polarity combinations, split/material leakage, sampler visits, mixed-truth shortcut rejection, effective CE microbatch equivalence, resume, position baselines and acceptance guards are covered. Both runtime workers pass CPU/CUDA parity, candidate chunking/permutation, numeric IDs and JSON. Repeated-inference memory is 164 MiB at all five boundaries. Pilot validation-boundary ranges are 650–696 / 652–690 MiB; all logged updates use CUDA, no observed fallback. Candidate sampled CE mean falls 0.8441 → 0.5500; held-out transfer does not improve sustainably. Explicit unknown visits rise to 6,340 EN / 6,460 ZH, without solving joint factual behavior. Chart: `docs/images/decision-v03-robust-comparison.png`. Logs: `tmp/robust-v03-r2-run.log`, `tmp/robust-v03-{tests-final,lint-final,learning-tests}.log`.

Next requires a new plan and independent panel. The bounded result motivates a foundational-representation comparison, not another unbounded supervision sweep; a public pretrained encoder remains separately authorized Path C. This package comparison does not isolate the cause or prove size/data/compute alone will fix it. The user removed Path B (Qwen semantic preprocessing); do not resume it from historical distillation records. RL, architecture growth and business integration remain outside scope.

User follow-up (2026-10-03): **training-data quality must be audited first in the next round**, before choosing a representation/capacity change. See the next-round prerequisite in [v03.md](v03.md): target-blind stratified review and complete contrast-family checks, annotation/answerability compatibility across public sources, candidate/translation semantics, shortcut/duplicate/diversity audits, subtype fitting/exposure, and a bounded same-parent quality comparison with newly reserved reviewed evaluation. Do not infer sufficient quality from generated gold correctness or row counts, and do not assert that bad data is already the proven cause. No new experiment is started by this follow-up.

## Completed goal — Decision v0.2 factual scoring (2026-10-02)

The user authorized [v02-plan.md](v02-plan.md); the bounded iteration is complete. See [v02.md](v02.md) for results, ASCII hierarchy, chart, use/reproduction and lessons. Ruby/Torch.rb, own weights, one shared bilingual model, unchanged 6.63M sinusoidal architecture, GPU-first 4,096 MiB process budget, model-only scope. **v0.1 remains the delivered preview; no v0.2 checkpoint is promoted.**

Artifacts: `runs/decision/factual-v02-r2/`; runner `benchmarks/decision/factual.rb`. The 128-row fitting check passes at updates 100/200. Both CE and margin finish 2,000 CUDA updates, **identical 64,000 row visits and 9,329,604 valid tokens**; GPU boundary memory 650–696 MiB, no fallback. No checkpoint passes the per-language routing guard, so seed-2027 confirmation is skipped. Calibration and **one-time** clean acceptance finish; `acceptance-opened.json` prevents reopening for tuning.

Acceptance factual macro: v0.1 **46.33%**, own broad parent **47.09%**, CE **67.33%**, margin **66.47%**. CE pairs: **64.53% English / 99.92% Chinese**; its English actor-binding pairs are only **0.31%**, Chinese unknown accuracy **0%**. Routing falls to **68.38% / 70.40%**, versus v0.1's **78.43% / 81.09%** on the already observed regression panel. Both final checkpoints pass CPU/CUDA probability parity and candidate-order smoke checks, but these engineering successes do not repair the behavioral failures. The margin has no supported advantage and is not the recommended default.

Preserve the original panels/protocol. The original state-only historical filter misses connected DuReader answers: four validation rows belong to seen components. A pre-prediction whole-component audit excludes 12 calibration and 26 acceptance rows; `provenance.json` pins `calibration-clean.jsonl` (820) and `acceptance.jsonl` (7,144), plus group exclusions/hashes. Frozen validation is development material, not a fresh acceptance claim. Future preparation excludes complete historical components from the outset. Also preserve the rejected oversized preparation and original generator fitting run (`factual-v02-rejected-budget/`, `factual-v02/`); no acceptance was opened there.

Read `report.json`, `controlled_slices`, source/language cells, group-bootstrap intervals, `diagnostics-trained.json`, exposure and raw/calibrated metrics before drawing conclusions. Diagnostic CE load path: `runs/decision/factual-v02-r2/ce-1337/choice`; Predictor loads temperature 1, while `calibration.json` separately records the evaluated temperature. Delivered API remains `Release.load("runs/decision/v0.1-preview", device: :auto)`. No new business API is needed.

Next task priorities: make initializer choice retention-aware (start with the own delivered v0.1 parent); factor queried actor and fact order independently, including question flips on an unchanged state; balance recorded unknown/known supervision per language and test reviewed candidate-wording coverage. Chinese unknown receives 1,143 visits versus English 2,271 in the current mixed bucket—an exposure imbalance, not a proven cause. Keep CE as the reference, and reserve a new acceptance panel before further optimization. Current acceptance is now regression-only; do not repeatedly tune on it or grow the architecture before these specific failures are addressed. A further optimization is a new bounded task, not unfinished work in this goal.

Verification: **192 production tests / 2,250 assertions**, zero failures/errors/skips; factual tests 8/92 and benchmark tests 7/23. Full lint checks 175 files clean, final report changes separately clean, whitespace check passes. The reviewed chart is `docs/images/decision-v02-factual-comparison.png`. Completed v0.2 work and the RL learning roadmap were subsequently committed as `22293ca`. Replaying `all` repeats fixed controlled allocations as regression; a different output path alone does not reserve new controlled acceptance.

## Completed delivery goal — Decision v0.1 (2026-10-01)

The user authorized implementing v0.1 with a reasonable measured result and clarified model-only scope; no easy_biz/business integration. Bilingual request-domain scoring is the explicitly stated initial supported-profile assumption, retaining arbitrary-candidate probability scoring. See [v01.md](v01.md). This goal takes precedence over earlier open-ended experiments. **v0.1 scoring-preview delivery is complete. User approved this scope and clarified 80% is advisory. The portable preview is published and verified, preserving the historical failed strict acceptance.** No new commit requested; preserve existing staged work.

First fit `runs/decision/release-v01-routing`: 2,400 CUDA updates, early stop, selected step 1,200 by NLL .75694. Independent original official-test panel: 800 groups per language. English/Chinese overall accuracy 79.5%/82.625%; confidence-selected 87.463%/88.825% at coverage 84.75%/87.25%. Failed the frozen release criteria. Runtime passes; original 16-candidate chunks gave about 74/102 ms CPU/CUDA on two warm validation examples. No CPU fallback; training validation memory plateau 646 MiB. First overstrict development exclusion audit is preserved: `tmp/release-v01-prepare-rejected-history.log`.

Bounded correction `runs/decision/release-v01-corrective`: own selected initializer, same 18,377 training rows, same validation/calibration, natural within-language sampling rather than class oversampling, LR .00005, at most 800 updates. Stopped at 700; selected 300 by validation NLL .691998. Prospectively fixed calibration guard: .94 selected point accuracy and .90 Wilson lower bound; acceptance gates remain unchanged. Reserved 400 NEW groups per language from 881 previously unused official TRAIN groups, not the opened first test. All historical prepared material/parallel IDs excluded, original partition retained. Prepared rows 408/402; independent per-language units exactly 400. Calibration reuses development data and is not claimed untouched. `protocol.json` remains original; `provenance-supplement.json` explains an inherited historical exclusions field, corrected in future preparation.

Corrected independent acceptance: English overall **78.25%**, balanced **73.185%**, selected **90.323%**, coverage **77.5%**, selected Wilson lower **86.521%**. Chinese overall **81.0%**, balanced **76.665%**, selected **91.641%**, coverage **80.75%**, lower **88.111%**. Chinese and both languages' confidence gates pass. English misses full accuracy 80% by 7/400 examples. These panels differ, so first/corrected test accuracies are not a paired model comparison. Both opened panels are now regression-only.

Delivery choice resolved on 2026-10-01: the user chose the scoring preview and clarified that 80% is a soft direction while optimizing step by step. Portable bundle: `runs/decision/v0.1-preview`, load with `EasyAI::Decision::Release.load(..., device: "auto")`. Explicit metadata `status: preview`, `acceptance_passed: false`; inference `release_status: preview`. Historical gates/results remain unchanged. Strict `Release.publish` still rejects failed acceptance; `preview: true` preserves it after provenance/runtime checks. Reproduce packaging with `benchmarks/decision/release.rb --phase preview --output runs/decision/release-v01-corrective`; destination must be new. Temperature 1.513142313469446, weights SHA `a143ceacc22ae0d0184de7452554c68a7bfd71f029ef68d15f5dcd909234c135`. General semantic improvement remains future work, not a v0.1 claim.

Corrected runtime checks pass: CPU/CUDA max probability delta <2.5e-8, reversal deltas zero, one 18-candidate chunk, published-bundle packaging rerun mean 54.3/58.4 ms CPU/CUDA, boundary GPU memory stable 202 MiB. Latency is a two-example warm-cache smoke observation, not a production benchmark. No training processes remain. Published-wrapper smoke passes CPU/CUDA, JSON, numeric IDs and reordered candidates; Chinese alarm example is correct, English “Wake me up at seven tomorrow” incorrectly prefers calendar (50.53%) and requests review. Preserve this failure in subsequent regressions. Strict stable packaging rejects unmet acceptance; explicit preview packaging is authorized. Formal files: `candidate_trainer.rb`, `release_metrics.rb`, `release_policy.rb`, `release.rb`, `data/routing_corpus.rb`; delivery runners: `benchmarks/decision/release{,_corrective}.rb`. Public JSON remains Ruby-assembled. No teacher weights, RL, keyword inference or business action code.

Final preview verification: **177 tests / 2134 assertions**, no failures/errors; learning **7 / 14**; RuboCop **152 files**, clean; whitespace clean. Logs `tmp/release-v01-preview-{tests,learning,lint}.log`; published-wrapper results `tmp/release-v01-preview-smoke.json`. Reviewed first/corrective loss figures tracked under `docs/images/decision-v01-{routing,corrective}-loss.png`; original learning figures preserved. Full tests and repository lint include the preview publisher and future-protocol provenance fields. Runtime CPU fallback and acceptance re-opening guards have regression tests. Next optimization should diagnose weak per-domain recall and generalization on validation, with a fresh prospectively reserved panel; do not automatically start a model-size/position/RL sweep.

## Proposed next iteration — factual scoring

See [v02-plan.md](v02-plan.md) for the historical chart audit, bounded CE-versus-pair-margin procedure and data independence. That iteration is completed, not active; follow the v0.2 results above and the accepted, not-started v0.3 plan for future work. Corrective training means first/last 100 updates are 0.344/0.241; validation reaches 0.692 at update 300 then plateaus. Do not mistake batch noise for absent gradients or domain classification for truth-scoring competence.

## Post-delivery feedback — multilingual API and negation (2026-10-01)

User wants one model handling multilingual input automatically and reports failures on negation. The model already shares weights/tokenizer across English/Chinese. Made `Release#probabilities` locale optional; it affects profile-specific review policy only, not neural scores. `route` still requires locale to render its fixed texts. Tests 6/25 and targeted lint clean. No weights changed or new fit launched.

Published-checkpoint CPU contrast diagnostic: four pairs per language, eight individual examples each, 4/8 correct and **zero fully correct pairs** in both. All prefer affirmative answers even for negative states. Full cases, options and probabilities are recorded in [v01.md](v01.md); raw script/report `tmp/decision-v01-negation-check.{rb,json}`. These observed examples are regression material, not fresh evaluation. Next prioritize truth/negation supervision and actor binding on the shared scorer, with varied domains/languages and a prospectively reserved held-out panel, while measuring routing preservation. Routing accuracy is not general truth-scoring accuracy. Do not implement negation keywords at inference or silently promote other languages as validated.

## Current handover — natural-task pilot completed (2026-10-01)

The authorized next iteration is complete: four seed-1337, fixed-1,000-update CUDA fits comparing old/broad data and sinusoidal/RoPE, followed by calibration and five evaluations including the unchanged own parent. No CPU fallback. See [natural.md](natural.md) for the exact protocol, tables, charts, interpretation, reproduction command and proposed next diagnostic. No checkpoint is promoted as reliable, and no new commit was requested. Existing staged fitting edits are preserved.

Main fresh source-macro accuracy: parent 42.43%; old sinusoidal/RoPE 43.65%/43.54%; broad 61.04%/60.84%. Broader data gains 17.40/17.31 points over controls, concentrated in news (75.8–79.3%) and en/zh request domains (47–51%). QA/NLI is uneven: broad RoPE loses 5.41 points on OCNLI versus its control; broad sinusoidal DuReader is 4.67 points below parent. Withheld Emotion remains 6–7%, below chance/majority, with `surprise` predicted on 81.5%/75.25% despite 2.75% gold frequency. Binding group correctness falls to 5.73%/4.56% versus 7.36% parent. These outcomes do not pass broad semantic/preservation gates. Tiny positional differences establish no RoPE superiority; retain sinusoidal default.

Artifacts: `runs/decision/natural-v1-pilot/{protocol.json,report.json,summary.txt,provenance-supplement.json}`; per-condition `{selected,calibrated,predictions-*.jsonl,coverage.json,report/}`; `evaluation-*.json`. Main panel has 1,269 supported-length decisions; independent held-out Emotion has 400. All are now observed diagnostics, not fresh acceptance for future retuning. Raw data, weights, logs and runs remain ignored. Reviewed figures are tracked as `docs/images/decision-natural-{comparison,loss}.png`; original learning plots remain in README.

The 226,900-row broad corpus is not fully consumed: 32,000 visits/27,549 unique rows; news sees only 6,325 of 101,343 training rows. Full-budget row visits match exactly within both positional pairs. Correct reconstructed tokens: controls 3,255,350; broad 4,939,843. Selected controls are step 200 (6,400 visits/640,992 tokens), selected broad step 1,000 (32,000/4,939,843). Equal allocated budgets do not mean equal evaluated-checkpoint exposure. Grouping initially undercounted tokens and the separate `choice_loss` component in the two controls; total backpropagated/validated/charted CE was correct. The corrected hook/components pass loss/gradient/counter regressions; original raw control traces are preserved and reconstructed exposure is in the report.

Preparation rejected an initial material overlap (generic short QA answers shared across question-conditioned splits); preserved root `natural-v1-pilot-rejected-overlap`. Current common panels filter training material and enforce disjoint groups/material. Dense microbatch 4 settled at 2,914 MiB; eight exceeded 4,096 budget, sixteen OOM. Actual validation boundaries plateau at 570/598/692/726 MiB, not allocator peaks. The original frozen manifest omitted raw semantic-source hashes. A post-pilot supplement verifies all six current raw files against the older parent manifest and rebuilds all 501 fresh public decisions identically. Future preparation verifies/pins these sources explicitly; original pilot protocol was not rewritten.

Permanent runners: `benchmarks/decision/{natural,natural_evaluation,natural_memory_profile}.rb`; formal schema/split code: `lib/easy_ai/decision/data/{natural_adapter,natural_corpus}.rb`; CSV is an explicit gem for Ruby 3.4. Training image code remains `lib/easy_ai/decision/training_report.rb`, with cross-condition plots in `natural_evaluation.rb`. One-command reproduction requires the own parent, earlier semantic/MASSIVE/Emotion caches and NVIDIA access; use a new output root.

Next: isolate unseen-option priors with unchanged-checkpoint candidate-only/neutral-state/wording diagnostics; do not insert semantic keyword inference rules. Then consider a controlled train-only MLM warm-up plus the same CE mixture versus CE continuation, from identical own tensors, fixed size/positions, per-task accuracy and probability preservation checks, and a prospectively reserved new held-out task family. This is a hypothesis, not a proven fix. Broad validation is still descending at step 1,000, so more CE can improve trained tasks, but does not guarantee unseen semantics. Do not automatically launch the old larger three-seed/size plan as if this pilot passed all gates. No teacher weights or RL required.

Verification: 156 tests / 2067 assertions passed; learning 7 / 14 passed. All run budgets, initial tensor equality, row visits, memory traces, SHA-pinned data and source rebuild are recorded. Final RuboCop: 140 files, no offenses; staged and unstaged whitespace checks are clean. Logs: `tmp/natural-*.log` (training/evaluation, counter tests, provenance rebuild, tests and lint).

## Previous completed round — fitting (2026-10-01)

User authorized the recommended next experiment. Implemented `benchmarks/decision/fitting.rb`: seed-1337 2x2 parent/random-start x sinusoidal/RoPE, 2,000 updates each, same initial tensors per pair, full-candidate deterministic permutations, dropout/evidence off. See [fitting.md](fitting.md) for the protocol. Data audit: 128 unique inputs, no conflicting labels, 64 rows per language, 32 distinct states and token sequences.

Caught and repaired a new harness issue: Torch.rb same-device `Module#to` replaced live parameters after optimizer creation. The evaluator now preserves already placed parameters. CPU regression and a real CUDA optimizer-update check passed. Invalid first attempt is preserved in `runs/decision/fitting-v1-invalid-evaluator/` and excluded from results. Historical evidence training's internal validation does not use that parameter-replacing evaluator.

All four valid 2,000-update runs finished on CUDA under `runs/decision/fitting-v1`. Parent sinusoidal/RoPE both reach 100%, first fitting gates at updates 1,800/1,000. Random sinusoidal/RoPE finish at 75%/87.5%; random RoPE fits Chinese fully but English remains 75%. Exact initial tensor equality, identical row-visit vectors and zero candidate-permutation logit errors verified. Reports, failures, checkpoints and charts are preserved; a reviewed chart is copied to `docs/images/decision-fitting-comparison.png`. Do not promote diagnostic fitting as generalization. At fitting completion, the broader round had not started; its completed results are now recorded in the current entry above. Continue from our own scratch-trained parent, test independent natural/relationship transfer, retain a sinusoidal reference and common selection/calibration panels; do not infer broad improvement from this fit. Cold-start English binding remains a separate unresolved optimization diagnostic. Logs: `tmp/fitting-*-valid.log`, final `tmp/fitting-report.log`, `tmp/fitting-cuda-regression.log`, `tmp/fitting-input-audit.log`. Preserve initialized checkpoints/protocol and failed attempts. Verification: 142 tests / 1918 assertions and learning 7 / 14 passed; RuboCop: 133 files, no offenses; `git diff --check`: clean. All four training traces contain exactly 2,000 CUDA updates and 64,000 row visits. No commit requested for this follow-up.

## Previous completed round — evidence (2026-10-01)

Resumed at the user's request. The evidence round and its supplemental evaluation tooling are complete. No additional training was launched during this review, and no checkpoint is promoted as reliable. Existing staged edits were preserved; the completed changes are being committed at the user's request.

Completed and verified:

- Optional evidence head, sentence/token alignment, paired CE/CE+evidence experiment and conditional-parameter AdamW exact-resume fix.
- Both 300-update capacity checks, all six 800-update main CUDA fits and all nine evaluations on each of the primary, natural-task and shared-benchmark panels. All main selected checkpoints remain step 100 by validation NLL.
- No meaningful answer improvement from evidence supervision: novel accuracy 50.71% vs 50.76%, binding 7.60% vs 7.60%. Natural transfer is weak. Shared full-denominator accuracy is about 19–20% / 8.6–8.9%, with context coverage 58% / 27.5%. See [evidence.md](evidence.md).
- Shared adapter/scorer/download review and 12 regression tests. Invalid vectors count as failures; correctness is recomputed from predictions. Replaying all nine stored prediction files preserves every correctness and support count.
- Seven raw snapshot checksums verified from cache; rebuilding the 2,231-decision panel produces byte-identical data. CLI supports download/prepare/evaluate/report/all; absolute-path invocation is tested. HF service response snapshots are byte-pinned, not revision-pinned; changed response bytes must not be silently accepted.
- Full suite: **137 tests / 1790 assertions**, zero failures/errors; learning: **7 tests / 14 assertions**, zero failures/errors. RuboCop: 131 files, no offenses. `git diff --check`: clean.
- README and retrospective synchronized; reviewed comparison/loss charts copied into tracked `docs/images/`. Existing learning illustrations retained.

Artifacts:

- `runs/decision/evidence-v1/report.json`, `generalization/{report.json,summary.txt}`, `shared-benchmarks/report.json`, per-run reports/predictions/checkpoints: gitignored.
- Raw files: `data/decision/downloads/shared-benchmarks/jev-{original,easy,hard}.jsonl`, `typed-{000,001,002,003}.json`: gitignored.
- Permanent runners: `benchmarks/decision/{evidence,evidence_evaluation,generalization,shared_benchmarks}.rb`; formal schema adapter: `lib/easy_ai/decision/data/benchmark_adapter.rb`.
- Review logs: `tmp/resume-{full-tests,learning-tests,lint}.log`, `tmp/shared-benchmarks-{cache,score-audit,prepare-audit,tests}.log`.

Next implementation: follow [next-experiment.md](next-experiment.md). First instrument and resolve the incomplete 128-row answer fit. Then freeze broader natural-task training and independent held-out evaluation, with a common validation panel across paired controls and real CUDA context-memory profiling. Keep shared test cases evaluation-only; all previously inspected tests are now regression diagnostics. Do not introduce teacher weights, keyword inference rules or model scaling to conceal an unresolved fitting failure. The larger-round protocol is proposed, not an executed result.

For the existing completed panel:

```bash
bundle exec ruby benchmarks/decision/shared_benchmarks.rb --phase download
bundle exec ruby benchmarks/decision/shared_benchmarks.rb --phase report --output runs/decision/evidence-v1
```

## Historical handover entries

The entries below record earlier stages. Their “next” instructions and test counts are historical; the current state above takes precedence.

## 1. 接手时的约束与状态

- 保持 Ruby / Torch.rb 路线：Ruby 实现算法和训练流程，LibTorch/CUDA 执行张量计算；当前选择从零训练，不导入外部基础权重或教师输出。
- 本机 Ruby 3.4.5、Torch.rb 0.23.0，Quadro RTX 3000、6 GiB。GPU 优先，进程显存软预算 4096 MiB；容量不足时缩小批次/分块，再回退 CPU。当前实现 FP32。
- 第一项产品能力是输入 `state / question / options`，输出候选概率。候选 ID 可为数字或字符串，输出键统一为字符串。JSON 由 Ruby 组装，网络不生成 JSON 文本。
- 语言目标不局限英语；本轮关系与公开语义实验是中文、英文，不能因此声称已有任意语言能力。
- 正式能力在 `lib/easy_ai/`；历史教学代码在 `learning/`；实验编排在 `benchmarks/decision/`。配置每行一个参数，README 与旧 learning 曲线继续保留。
- `data/`、`runs/`、下载文件、权重被 Git 忽略。文档和精选图表在 Git 中。只 clone 仓库**不会得到本机训练产物**。
- 交接基线验收：正式测试 71 tests / 640 assertions，教学 7 tests / 14 assertions，RuboCop 95 files，全通过。后续代码的最新验收记录见 [人物绑定对照](binding.md)，不能用基线结果代替新修改的验证。

已完成：数据准备、MLM/候选训练、checkpoint/续训、校准、CPU/CUDA 推理、RoPE、两种池化和联合编码对照、成对评估、课程实验、训练图表与显存累积修复。**当前所有课程种子仍未通过泛化门槛。**

## 2. 原诊断权重与可复现入口

本机项目根目录：`/home/andersen/as_projects/AI/easy_ai`。以下命令均从项目根目录运行。

按 [README 的环境说明](../../README.md) 安装依赖后，可用 `bundle exec irb` 执行：

```ruby
require "easy_ai"

predictor = EasyAI::Decision::Predictor.load(
  "runs/decision/relations-v2-curriculum/choice/best",
  device: "auto"
)
```

这是 seed 1337 的课程模型：自训 sanity 1000 步，再训练完整数据；选中第二阶段第 800 步。末尾 `/best` 选择最佳 validation 权重，直接加载 `/choice` 选择最后权重。它未校准，`calibrated: false`。

本次诊断固定产物：

```text
runs/decision/relations-v2-curriculum/choice/best/checkpoints/step-00000800-75247cec
weights.pt SHA256:
b1f4aa46048d045a9b98baae1ef6d15ec779406a49ef72657ff57c6e5baac3dc
tokenizer fingerprint:
3e28d05d3a2711828a45a8c77994285b5350ecfb1bf248289f45aa182888fa3c
```

模型 6,627,841 参数，hidden 256、4 层 encoder、4 heads、FFN 768、1 层 cross-attention，`rotary / all / separate / matching`，dropout 0。embedding 容量 12000，本轮专用 tokenizer 实际为 400。**读取 checkpoint 的有效配置，不要将 `relations.yml` 的默认 sinusoidal 当成这个权重的配置。**

本地数据：`data/decision/relations-v2/`。train 13824 条、108 个家庭；validation / calibration / test 各 1280 条、20 个家庭。sanity 是 train 中一个家庭的 64 条。test-familiar 与 test 共用家庭，仅句式不同；不能当成两份独立测试证据。

当前代码生成 v3 数据（输入文本/标签与 v2 相同，增加分组元数据；输出目录必须不存在，不覆盖历史 v2）：

```bash
bundle exec ruby bin/easy-ai prepare-relations --output data/decision/relations-v3 --vocab-size 400 --seed 1337
```

一条命令重训当前课程基线，自动生成新 run 目录、日志与图表：

```bash
bundle exec ruby benchmarks/decision/relations.rb --variants rotary --seeds 1337,2027,3407 --curriculum --patience 0 --steps 2000
```

每个 seed 是 1000+2000 步，第二阶段重置优化器和 warmup。新机器重训后的 checkpoint 目录带新后缀，以新 run 的 `summary.json` 为准，不会重建上述同名路径或承诺跨设备逐 bit 一致。没有 `--curriculum` 时，完整阶段重新随机初始化。

单独评估已有最佳权重：

```bash
bundle exec ruby bin/easy-ai evaluate-relations --checkpoint runs/decision/relations-v2-curriculum/choice/best --data data/decision/relations-v3/validation.jsonl --batch-size 128 --controls
```

当前结果汇总：`runs/decision/relations-v2-comparison/index.html`。其他课程 seed 在 `runs/decision/relations-v2-curriculum-2027/`、`runs/decision/relations-v2-curriculum-3407/`。详细拆分、全部对照、图表见 [关系实验](relations.md)。上一轮全量审计曾使用 `/tmp` 临时脚本；接手不依赖该脚本，使用已入库的实验入口和 `evaluate-relations`。

跨机器交接时，另外复制 `data/decision/relations-v2/` 与所需 run 的完整目录，保留配置、日志、报告和 checkpoint 指针。只做本次推理复测，也可复制上述固定 checkpoint 的整个目录，并把加载路径改为该目录；不要只复制 `weights.pt`，加载还依赖 `tokenizer.json`、`metadata.json`、`manifest.json` 及 manifest 列出的其他文件。续训需要含优化器状态的 checkpoint，不能将只有推理权重的产物当作完整续训状态。复制后核对上面的权重 SHA256；缺少产物时按重训命令生成新基线，并记录它与本次固定权重的区别。

## 3. 用户复测与已核实证据

### 英文并没有判断正确

用户将以下英文输出描述为正确，但实际选项是 `0 = will`、`1 = no`：

| state | 模型 argmax | 对文本表达方向的判断 |
| --- | --- | --- |
| `I think i will be late` | `no`，80.12% | 相反 |
| `Time is enough , I will not  be late` | `will`，51.24% | 相反 |
| `我要迟到啦！` | `不是`，55.72% | 相反 |
| `还早，不会迟到啦。` | `不是`，57.50% | 方向正确 |

前两条 question 为 `will I late?`，中文 question 为 `是不是要迟到了？`。这些是用户报告的输出，不是新增 benchmark 成绩。`I think` 的语义涉及主观判断，后续须区分“说话者表达将迟到”与“现实中一定迟到”。不能用语法不标准解释下方标准训练句也失败的情况。

### 买票例子就在训练集中

以下两个 state，配相同问题 `以下说法成立吗：小周买了票。`，都存在于 `data/decision/relations-v2/train.jsonl`，家庭为 `[0, 1, 0]` / `relation:0:1:0`：

| state | 训练 target | 用户 P(成立) |
| --- | --- | ---: |
| 小林买了票，小周没有买票。 | `no` / 不成立 | 0.3618226686 |
| 小周买了票，小林没有买票。 | `yes` / 成立 | 0.3933046863 |

训练中的候选排列为 `no, yes`，用户排列为 `成立, 不成立`，且 ID 为 `0, 1`。ID 不进入网络；应按候选文本比较概率，不能按数组下标解释为标签反转。

训练集成员检查可独立运行：

```bash
bundle exec ruby -rjson <<'RUBY'
states = ["小林买了票，小周没有买票。", "小周买了票，小林没有买票。"]
File.foreach("data/decision/relations-v2/train.jsonl") do |line|
  row = JSON.parse(line)
  next unless states.include?(row["state"]) && row["question"] == "以下说法成立吗：小周买了票。"
  puts JSON.pretty_generate(row.slice("id", "group_id", "state", "question", "options", "target"))
end
RUBY
```

这两条应作为训练回归样本，不能称为独立泛化测试。原来“主要是没见过表达”的说明不足以解释本次错误。

### GPU 复测：主体变化的响应很弱

固定上述权重，候选为 `0: 成立 / 1: 不成立`；每条问题都是 `以下说法成立吗：<人物>买了票。`：

| state | P(小林买了票成立) | P(小周买了票成立) |
| --- | ---: | ---: |
| 小林买了票，小周没有买票。 | 36.74% | 36.18% |
| 小周买了票，小林没有买票。 | 40.01% | 39.33% |
| 小周没有买票，小林买了票。 | 44.73% | 43.75% |
| 小林没有买票，小周买了票。 | 49.51% | 48.34% |

四条 state 的编码序列互不相同；每条请求在清空 state cache 前后概率差值均为 0。这个复测排除了这些请求上的分词碰撞和缓存复用错误，不代表证明所有运行时路径都没有 bug。模型对人物变化响应太弱，句序变化也带来明显概率漂移。

复测请求与清缓存方法：

```ruby
request = {
  state: "小周买了票，小林没有买票。",
  question: "以下说法成立吗：小周买了票。",
  options: [{ id: 0, text: "成立" }, { id: 1, text: "不成立" }]
}
first = predictor.probabilities(**request)
predictor.clear_cache
second = predictor.probabilities(**request)
p [first, second]
```

## 4. 已知结果与尚未证实的解释

| 已测项目 | 结论 |
| --- | --- |
| 64 条训练拟合，seed 1337 | 原位置编码两种池化均 75%；RoPE 两种池化均 100% |
| RoPE 直接完整训练，3 seeds | 新句式 test 平均 69.90%，范围 62.58%–75.00% |
| RoPE 课程训练，3 seeds | 新句式 test 平均 80.03%，范围 78.91%–80.78%；仍未达标 |
| 直接延长至 6000 步，无早停 | seed 1337 最佳仍为第 1200 步，test 72.11% |
| RoPE 联合编码，2000 步 | test 74.22%，没有解决绑定；不能据此放弃状态缓存 |
| 当前最佳课程权重的 train | 总体 93.71%；同真假 100%，混合真假 87.43% |

三种子共享同一测试拆分；各实验预算不同，这不是严格等计算量的因果证明。当前权重的总体训练 accuracy 仍有约 6% 错误，不能将小规模 sanity 的 100% 误写成完整训练集全对。

交接基线的代码事实：`RelationCorpus` 把 `question_flip` 写入 `contrast_group`，`PairSampler` 每次将这两个样本一起采样。当时没有专门的主体切换/人物角色互换训练分组，评估包含 question_flip、fact_flip、irrelevant_fact、order 四类；v3 已扩展这些能力，见新实验记录。

待验证假设：现有配对更容易强化问题肯定/否定变化，而对“否定属于谁”的约束不足。全局池化可能弱化局部关系，但已有 cross-attention，不能把错误直接归因于“没有 attention”。单家庭热身、固定学习率、有限句式都可能影响训练；需要逐项对照。

## 5. 原始推进计划与顺序

本节保留交接时的任务定义。当前已实现 P0 的新行为指标、P1 的四条采样，并完成两策略三种子对照；还新增了 P2 的代表性热身集配置与试验。新的隔离挑战集、更细的多阶段课程、学习率调度、额外监督和公开语义混合仍未实现。逐轮实验结论以 [人物绑定对照](binding.md) 为准。

### P0：先把主体绑定变成明确的评估项

新增主体切换与角色互换检查，以 gold 逻辑决定关系，不把所有变化都强制标为翻转：

```text
                         问 A 买了票？  问 B 买了票？
A 买了票，B 没有买票           是            否
A 没有买票，B 买了票           否            是

两人事实真假相同：切换被问人物 -> 答案不变
两人事实真假不同：切换被问人物 -> 答案翻转
交换句序或修改无关事实         -> 答案不变
```

- 为中文、英文分别报告 accuracy、整组全对率、混合真假分项、变化方向、候选排列/分块一致性。需要分开识别“忽略人物”和“对句序敏感”。
- 用户买票例子已在 train，归回归检查；迟到探针归已知人工开发检查，不能调参后再称盲测。
- 保留 v2 及历史结果可复现。新数据写新版本目录和 manifest；相关翻转、翻译、句序、候选改写仍按语义家庭整体拆分。
- 旧 test 已多次查看，继续用于历史比较时注明探索性质；为下一轮最终验收预留新的隔离挑战集，在选择方案前固定拆分与门槛。

### P1：先做成组采样对照，保持交叉熵与模型不变

将上面的四条作为完整训练组，突出一真一假的情况，同时保留同真假样本，避免训练成另一种固定答案偏差。对照旧 question_flip 配对与新主体/角色分组；平衡语言、人物角色、标签和候选位置。记录每类实际采样量。

注意当前 `PairSampler` 强制每组恰好两条，config 只检查 microbatch 为偶数；不能只把 `contrast_group` 扩成四条。需要明确新分组元数据与采样器、批大小校验、显存缩批后的梯度累积行为，并保留旧两条模式的读取/续训兼容。训练分组和逻辑元数据不得送入网络作为文本输入。

### P2：扩展课程，单独比较优化设置

当前 sanity 只有 `[5, 7, 4]`：小张/小赵（Frank/Henry）与雨伞。新课程起点应覆盖全部动作与多个人物，先小规模完全拟合，再逐级增加家庭数、无关事实和表达变化：

```text
单个人物明确事实 -> 覆盖多个人物/动作的小集合
                 -> 两个人物混合真假 + 主体/角色对照
                 -> 更多组合、干扰事实、句序
                 -> 问题/候选表达多样化 -> 公开语义混合训练
```

保留前一级部分样本检查遗忘，以预先定义的开发集能力门槛推进课程。一次只比较一个因素：先采样，再课程覆盖，再学习率衰减/阶段学习率。记录总步数、examples/tokens seen、阶段起点、seed 和计算时间，不仅报告最后阶段步数。

阶段学习率是待实现项：现有 `--init` 不能同时传 `--config` 或 `--tokenizer`，只靠换 YAML 不会覆盖已加载模型。若新增训练参数覆盖或 scheduler，需保持模型/词表不被悄悄替换，保存课程/调度状态，并验证中断续训与显存回退后的行为。

### P3：仍失败时，分别比较额外监督和结构

先保留交叉熵基线，再独立比较成对排序或证据定位辅助任务。成对排序不能代替单例正确标签；恒定输出也可能满足某些“不变”关系，所以仍看整组全对率。

证据定位可由生成器给出被问人物对应的事实片段作为训练标签；推理时由神经网络定位，不能用手写人物/否定规则直接计算答案。若需结构对照，研究受问题控制的证据汇总，保留 token memory，而不是只增加层数或重复加入已有的 cross-attention。

### P4：基础绑定达标后，再扩大语言与任务覆盖

增加 `成立/不成立`、`是/不是`、`会/不会`、`true/false`、`yes/no` 等表达及相应问题，明确候选语义；同一源例的变体不得跨 split。区分文本中的判断、说话者态度、未来不确定性，不能将 neutral/unknown 一律映射为 false。

继续使用公开标注数据，沿用来源/分组隔离和许可记录，见 [公开语义数据](semantics.md)。关系数据与公开语义数据的 tokenizer ID 不同；混合训练须使用共同的 train-only tokenizer，并据此从零初始化，不能把 400-token 词表的权重直接解释为另一套 12000-token 词表。当前 paired sampling 与 source balancing 不兼容，混合采样策略也需显式设计。

RL 留给之后的多步编排任务。参数扩展需在充分优化后仍表现出容量限制时做单独实验；温度校准不会改变 argmax，不能修复主体忽略。

## 6. 验收与产物要求

- 新增行为测试要先在基线模型上重现失败；对数据逻辑、组完整性、split 隔离、候选置换、续训和显存回退做必要回归。
- 下一轮小规模、覆盖多人物多动作的训练拟合建议门槛 ≥99%；完整泛化延续每语言 accuracy ≥95%、各类成对/成组全对率 ≥90%，包含新增主体/角色对照。门槛在实验前冻结，不因为结果差而降低。
- 至少保留 seeds 1337、2027、3407 的全部结果。用 validation 选择 checkpoint/方案；test 不用于逐步选择学习率、课程或错误样例。不能只报最好 seed。
- 分别评估选中权重与最后权重的 train/validation；检查删除 state、删除 question 后的退化，以及同真假/混合真假差距。通用输出概率的校准另用 calibration split。
- 监控连续验证的 GPU/RSS/临时 Tensor 数量；区分预热分配器缓存与持续增长。已有内存修复不得回退。参见 [内存回归](memory.md)。
- 每轮保存有效 config、数据/词表指纹、父 checkpoint、采样策略、步数/token 预算、语言分项、失败样例与原始训练/验证曲线。更新 README 和实验文档，保留失败结果与旧 learning 展示。

代码检查入口：

```bash
bundle exec rake test
bundle exec rake test:learning
bundle exec rake lint
```

## 7. 接手需要读/改的文件

| 目的 | 文件 |
| --- | --- |
| 逻辑世界、标签、拆分、配对定义 | [relation_corpus.rb](../../lib/easy_ai/decision/data/relation_corpus.rb) |
| 示例元数据与训练采样 | [example.rb](../../lib/easy_ai/decision/data/example.rb)、[pair_sampler.rb](../../lib/easy_ai/decision/data/pair_sampler.rb) |
| 成对指标、分语言和真假模式 | [relation_evaluation.rb](../../lib/easy_ai/decision/relation_evaluation.rb) |
| 训练、学习率、缩批、恢复 | [trainer.rb](../../lib/easy_ai/decision/trainer.rb)、[config.rb](../../lib/easy_ai/decision/config.rb) |
| 阶段初始化与命令参数 | [cli.rb](../../lib/easy_ai/decision/cli.rb) |
| encoder、局部交互和最终池化/打分 | [choice_model.rb](../../lib/easy_ai/decision/choice_model.rb)、[interaction_block.rb](../../lib/easy_ai/decision/interaction_block.rb) |
| 推理、候选分块、状态缓存 | [predictor.rb](../../lib/easy_ai/decision/predictor.rb) |
| 实验编排和门槛 | [relations.rb](../../benchmarks/decision/relations.rb)、[relations.yml](../../config/decision/relations.yml) |
| 必须覆盖的回归 | [relation_test.rb](../../test/easy_ai/decision/relation_test.rb)、[training_test.rb](../../test/easy_ai/decision/training_test.rb)、[memory_test.rb](../../test/easy_ai/decision/memory_test.rb)、[rotary_test.rb](../../test/easy_ai/decision/rotary_test.rb) |

接手第一步：先读 [人物绑定对照](binding.md) 的最新结果，再核实权重与数据版本。本页保留原始失败证据和任务顺序，不应重复实现已经完成的新指标与采样器。
