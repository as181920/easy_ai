# Decision 迭代回顾：证据、失败与下一步选择

## Latest learning record: factual supervision and pair margin (2026-10-02)

The [completed v0.2 round](v02.md) compares identical own initialization/data/seed/exposure under CE and CE + signed pair margin: 64,000 visits, 9,329,604 tokens, 2,000 updates per arm. New factual supervision raises acceptance source/language macro from v0.1's 46.33% to **67.33% CE / 66.47% margin**. Chinese finite-world complete pairs reach 99.92%, but English actor binding stays at 0.31%, Chinese uncertainty collapses to 0%, and routing loses about ten points versus the delivered model. No checkpoint is promoted; no eligible validation improvement triggers a second seed.

Useful lessons retained:

- Small-set fit and loss descent are necessary diagnostics, not generalization evidence. The new 128-row fit passes twice; CE sampled-loss means drop 0.99707 to 0.31422, while severe held-out slices remain.
- An auxiliary objective can improve early checks without improving the final capability. The fixed margin slightly underperforms CE here; retain the simpler reference rather than attributing synthetic-data gains to the new loss. This one seed does not establish a universal negative result for margins.
- Choose a parent against all capabilities to retain. The broad parent's stronger factual probe masks much weaker routing before continuation; recovery versus that parent still fails retention versus v0.1.
- Factor question/actor/order independently. English binding fails while longer irrelevant-fact templates pass; question-removal sensitivity remains high. These are reasons to test shortcuts, not proof of a particular mechanism or a justification for keyword inference rules.
- Audit actual language/source/unknown exposure. Balanced language draws do not balance a rare subtype inside different-sized pools: Chinese unknown receives 1,143 visits versus English 2,271. Balance declared supervision and test the hypothesis instead of assuming corpus size alone explains the failure.
- Unused answer text does not make a shared-question component independent. The historical audit removes whole connected DuReader components before prediction and preserves the original manifest as evidence. Never silently rewrite provenance after evaluating outcomes.
- Separate finite-template facts, naturally annotated QA/NLI and old routing regression. Macro rates, complete pairs, per-language slices and whole-group intervals expose failures hidden by high row-weighted accuracy.
- Compute raw/calibrated metrics from the same logits; calibration changes probability quality, not argmax. API CPU/CUDA parity and stable memory confirm engineering behavior, not semantic correctness. Latency measured inside a large reporting heap is not a standalone deployment benchmark.

The next priorities are retention-aware own initialization and factorized binding/uncertainty supervision with a newly reserved panel. Architecture growth, external teachers and RL remain unmotivated first responses to these identified failures. v0.1 remains the delivered preview; experimental v0.2 load paths and all failed gates are preserved in the results document.

## Earlier learning record: natural-task pilot (2026-10-01)

The [completed pilot](natural.md) separates broader supervision from positional encoding using the same own parent and a single-seed 2x2 comparison. Main macro rises from parent 42.43% to broad 61.04%/60.84%, with +17.40/+17.31 points over positional controls. News reaches 75.8–79.3%; bilingual request domains 47–51%. This supports the concrete hypothesis that missing task coverage limits those learned classifiers. It does not establish unseen semantic competence: Emotion stays at 6–7%, some QA cells regress, and binding groups pass only 4.6–5.7%.

Useful lessons retained:

- A small-set positional fitting advantage does not establish natural-task superiority: broad sinusoidal/RoPE results are effectively similar here. Retain the reference rather than promote an architecture from one seed.
- Always evaluate a truly withheld task family and per-source cells. Aggregate gains can conceal regressions. Emotion's `surprise` dominates 75–82% of predictions despite 2.75% gold frequency; investigate candidate-score priors without coding semantic word rules or claiming the cause is already proven.
- Equal allocated budgets differ from selected-weight exposure: control selects step 200, broad step 1,000. Corpus size differs from observed data: only 6,325 of 101,343 news rows are visited. Report actual visits and tokens before attributing gains to scale.
- Grouping full candidate sets avoids expensive padding but must aggregate instrumentation as well as loss. A token/component undercount did not change the optimized CE; corrected gradients/loss/bookkeeping are tested and old counters are preserved with a reproducible recount.
- Generic short answers can cross question-conditioned partitions. A stronger material-only holdout rejected the first preparation; filter transparently, keep the rejected attempt, and record the changed split scope.
- Pin prepared panels before fitting, preserve original manifests, and supplement provenance honestly. Raw QA/NLI hashes omitted by the pilot were subsequently checked against the older parent inventory; rebuilding 501 fresh public rows matches exactly. This is a supplement, not a retroactively rewritten pretraining manifest.
- CUDA process memory should stabilize after warming; observed boundaries plateau at 570–726 MiB. Dense profiling selects a safe microbatch and does not constitute an allocator-peak measurement. Charts keep raw losses, explicitly label moving averages and preserve the original learning illustrations.

More CE updates are plausible for trained tasks because broad validation still decreases at the budget boundary. Generic semantics remains a separate hypothesis. The proposed next foundation ablation is train-only MLM warm-up plus CE versus CE continuation from identical own weights, with a new independent held-out family and per-task preservation checks. No new result is claimed for that proposal.


Earlier completed follow-up (2026-09-29): the [gold-supervised coverage round](coverage.md) measures actual exposure before attributing failures to insufficient data or capacity. Longer training improves mean challenge accuracy by 3.05 points but worsens raw probability metrics; wording expansion loses 1.97 points and fails its gate. Keyword-based semantic categorization was rejected and removed; the counters use declared metadata only. The teacher route remains deferred. Section 9 records the new lessons.

Follow-up: the first [reusable distillation implementation](../distillation/README.md) now supports teacher content, pseudo-label export and candidate-level losses. Historical results below remain unchanged; implementing the pipeline is separate from demonstrating improved student semantics.

The [first real Qwen teacher run](../distillation/teacher-v1.md) fit the local GPU but failed candidate-order stability (91.67% against 95%). This adds a teacher-quality failure to the learning record; no new student quality result is claimed.

The [task-specific prompt follow-up](../distillation/teacher-task-v1.md) tested whether explicit task definitions fixed that failure. They corrected some inspected cases but introduced more regressions: agreement fell to 85.42% and correctness in both orders to 35/48 from 39/48. The lesson is to preserve the full fixed comparison, including regressions, rather than validate a prompt only against its motivating errors. Both profiles failed; student training has not used their audit labels.

更新：2026-09-28。本文从初版候选模型回顾到 v3 人物绑定实验，只保留能帮助提出假设、设计对照和解释结果的经验。这里的“基础模型”指本项目的随机初始化网络与自训 MLM；截至本次记录，没有导入 Qwen/mmBERT 权重，也没有运行教师蒸馏。

**当前结论：工程流程已经完整，状态依赖和小任务拟合取得进展，但通用语义没有建立，人物绑定也未稳定通过验收。语义覆盖不足与优化不稳定同时存在，不能用其中一个解释全部错误。** 下文实测与未来建议分别标明；不同任务的 accuracy 不能连成一条“能力持续提升”的曲线。

## 1. 从模型结构到第一轮失败

初版采用共享双向 Transformer encoder，状态编码可复用，问题和候选通过 cross-attention 读取状态，最后给每个候选一个分数。它适合可变候选比较，不需要自回归生成或文本 sampler。结构示意见[算法与设计](architecture.md)。

```text
状态 -----------------> encoder -> token memory --------+
问题 + 候选 -----------> encoder -> cross-attention <----+
                                          |
                                pooling -> score_i
                                          |
                              softmax(all candidate scores)
                                          |
                              Ruby 按 ID 组装 Hash / JSON
```

JSON 合法性来自程序约束，语义正确性来自训练与评估；二者不能互相替代。硬标签也能训练概率：交叉熵降低正确候选的负对数概率，不要求数据附带人工小数概率。MLM 只有 token 上下文监督，不能代替问题与答案之间的监督。

默认初版是 11,484,929 参数；加入输入归一化和 matching 头的 MASSIVE 配置为 11,747,841。后来的公开语义/关系配置缩小 embedding 容量，合计 6,627,841；并非参数持续增大。关系 tokenizer 实际只有 400 个 token，embedding 表仍有 12000 行，未使用的行不代表额外语言知识。

## 2. 状态依赖：loss 下降，模型也可能没读题

以下数字的完整上下文与命令在[候选学习诊断](diagnostics.md)。

| 实验与假设 | 观察到的结果 | 能留下的经验 |
| --- | --- | --- |
| 初版：200 个独立源样本 × 五语言，MLM 100 步、候选 300 步 | 八候选训练 loss 的前/后 30 步均值 2.07595 → 1.72144；中文 validation accuracy 29.5%，打乱或清空 state 后仍 29.5%，所有选择不变；标签频率基线 30.0% | loss 下降与高于随机 accuracy 均不足以证明读取了状态；必须做输入消融和简单基线 |
| 位置向量可能淹没 token 信号：只将位置尺度改为 0.02 | 原位置 RMS 约为 token embedding 的 34.5 倍；缩放后预测仍有 100% 的状态打乱一致率 | 测到异常尺度值得排查，但合理的解释不等于有效的修复 |
| 再加入 embedding LayerNorm | 原始中文 accuracy 33.5%，打乱后 24.5%；原始 NLL 2.49626，仍差于频率基线 1.90706 | 输出开始依赖输入，不等于依赖方式正确 |
| 增至 2000 源分组、重训 8k BPE、动态负例与标签平衡、直接监督 1000 步 | 最佳权重中文 accuracy 20.5%，打乱 state 后不变 | 多一点数据和步数没有自动解决退化；同时改了多项配置，不能归因成“数据越多越差” |
| 显式 matching 特征 `[q, s, q*s, abs(q-s)]` 与改进训练配置 | 中文 validation 原始 76.0%，打乱 21.5%，常量 state 19.5%；五语言采样八候选 test 75.6% | 本任务上有效使用状态有了证据；但多个配置共同变化，收益不能全部归于 matching 头 |

matching 成功后的中文“来不及了要迟到了”，模型仍给“会迟到”约 34.25%。意图分类学得更好，不等于已经学会任意问句、否定关系和候选表述。MASSIVE 的 Intent 问题与自然问答不是同一任务。

## 3. 公开语义数据：有标签，不代表已经充分学过

改用 OCNLI、DuReader-YesNo、BoolQ 共 **107094 条中英训练记录**。来源分组隔离，训练 tokenizer 不读取 held-out 文本，不能把它们看作 107094 个完全独立的语义现象。任务、拆分、许可与报告在[从零语义训练](semantics.md)。

模型随机初始化，固定候选阶段 1000 步、有效 batch 32，对照是否额外先做 1000 步 MLM：

| 同一批 test | 直接监督 | MLM 后监督 |
| --- | ---: | ---: |
| BoolQ accuracy | 63.39% | 62.50% |
| DuReader-YesNo accuracy | 70.72% | 69.96% |
| OCNLI accuracy | 38.57% | 32.86% |
| 来源宏平均 accuracy | 57.56% | 55.11% |
| 整体 NLL | 0.81873 | 0.83251 |
| ECE | 0.03832 | 0.03385 |

MLM validation loss 从 7.8895 降到 6.5975，却未转化为更好的下游判断。只能得出“这次短程 MLM 不奏效”，不能推断预训练普遍无用；两组总计算量也不相等。

候选阶段共抽样约 **32000 行次**，小于训练行数，且按来源均衡、有放回采样。因此“大数据已经下载”与“模型充分学习过数据”是两件事。尚缺独立源覆盖率、实际 token 数和各来源重复次数的完整审计。DuReader 的打乱 question 影响很小、OCNLI 的原始 state 优势不明显，提示仍有表面线索捷径。

概率校准不能修复错选：正温度不改变 argmax；ECE 降低也不意味着语义提高。先检查任务行为，再报告校准效果。

## 4. 显存问题：回退机制不能掩盖生命周期错误

长文本 validation 曾累积到约 4.95 GiB 并回退 CPU。原因是 Ruby 的小型张量包装对象背后持有大量原生内存，GC 未及时触发；`no_grad` 不等于释放张量。

修复把每个 batch/chunk 的临时张量限制在独立方法内，只返回 Ruby 数值，并在边界回收。连续五轮候选/MLM 验证预热后均为 1178 MiB；后续 MLM 1000 + 监督 1000 步分别稳定在 1220 / 1602 MiB，没有回退。完整测量见[内存与显存](memory.md)。

经验是区分活跃张量累积、正常 allocator 缓存和单个 batch 超容量。显存不是越接近满载越好；记录吞吐、验证耗时和占用稳定性，比只看 GPU 利用率更有诊断价值。记录的是进程采样占用，不冒充瞬时峰值。

## 5. 受控关系任务：先定位最小失败条件

从公开语言缩小到“谁做了什么、谁没有做”的两人任务，用生成器产生可验证标签，但神经推理不调用生成器逻辑。按人物/动作家庭隔离拆分，train 13824 行/108 家庭，sanity 是其中 1 家庭/64 行。v2 限制 tokenizer 为 400，避免极小语料把中文整句合并成近乎一个 token。详见[关系实验](relations.md)。

| 对照 | 结果 | 解释边界 |
| --- | --- | --- |
| 相同初始化、1000 步，正弦位置的 all / candidate 池化 | 两者中英训练拟合均为 75% | 只改池化未解除障碍；真假相同易、混合真假随机猜即可得到 75% |
| 相同预算，RoPE 的 all / candidate 池化 | 两者均 100% 拟合，删除状态/问题回到 50% | 此小训练集的优化障碍得到缓解，不代表未见语言泛化 |
| RoPE 全量随机初始化训练，三个种子 | 新句式 test 72.11%、62.58%、75.00%，平均 69.90% | 64 条记住了，完整人物绑定仍未学稳 |
| seed 1337 延长到 6000 步、关闭早停 | 最佳 validation 仍在 1200 步，test 仍 72.11% | 单纯延长这条优化轨迹没有改善；不能据此认定模型必然太小 |
| RoPE 联合编码 | sanity 100%，完整 test 74.22% | 更多输入交互没有自动解决绑定，也不足以支持牺牲状态缓存 |
| 先拟合 64 条，再初始化完整训练 | 三种子 test 80.78%、80.39%、78.91%，平均 80.03% | 训练顺序有改善线索；课程总预算更多，尚非完全匹配计算量的因果证明 |

旧课程权重在两条**已经存在于 train** 的买票样本上都偏向“不成立”。清空状态缓存前后无变化，分词后的 state 也不同。这类失败不能仅归因于缺少自然语言预训练；模型连已标注的关系也未稳定拟合。此时仍需检查监督、采样、优化与归纳偏置。

## 6. v3 四条成组与扩大热身：保留改善，也保留退步

v3 只加评估/采样元数据，文本、标签、顺序和 tokenizer 与 v2 一致。新增主体切换、角色互换、四条组全对率；元数据不进入网络。四条组仍是逐样本 CE，改变同次更新中的反例组合，没有新增知识或逻辑约束。完整实验与加载入口在[人物绑定对照](binding.md)。

P1 固定相同模型、数据、1000+2000 步、三种子，对比两条问题翻转采样和四条绑定采样：

| seed | 两条 test | 四条 test | 两条混合组全对 | 四条混合组全对 |
| --- | ---: | ---: | ---: | ---: |
| 1337 | 80.78% | 88.91% | 38.13% | 71.25% |
| 2027 | 80.39% | 82.34% | 56.25% | 40.00% |
| 3407 | 78.91% | 76.41% | 7.50% | 33.75% |
| 平均 | 80.03% | 82.55% | 33.96% | 48.33% |

平均 train accuracy 为 93.09% → 96.60%，但部分种子的 test 或混合组指标退步，六次都没过门槛。seed 1337 修复了训练内两条买票检查，仍会错在中英文“不会迟到”。单个高概率的正确例子不构成语义能力证据。

![P1 全种子对照](../images/decision-binding-comparison.png)

P2 将热身从 1 家庭/64 行改为覆盖全部人物与动作的 8 家庭/512 行，保持其余数据和预算不变：

| seed | 热身拟合 | 完整 train | test | test 混合组全对 |
| --- | ---: | ---: | ---: | ---: |
| 1337 | 93.55%，未达 99% | 未运行 | 未运行 | 未运行 |
| 2027 | 100% | 91.60% | 76.09% | 10.00% |
| 3407 | 100% | 99.57% | 89.61% | 79.38% |

不能丢掉第一个失败再报告“三种子平均”。两个初始化可以拟合 512 条，一个不能在给定预算内做到；更多覆盖没有自动稳定训练。第三个种子的 train/test 差距又显示泛化问题。至此，扩大数据覆盖、优化稳定性和模型容量必须分开检验。

## 7. 语义不足与 Qwen 蒸馏：下一步建议，尚未实施

用户提出“是否基础语义还没学会，所以后面的判断学不好”，这个疑问符合现有证据。关系数据只有少数人物、动作和模板，公开语义训练也很短，不能支撑开放表达、情态、时间、常识与多语言泛化。不过这些实验还不足以量化语义覆盖、优化和容量各占多少影响。

理解语言与学习判断可以联合发生，不要求先造出一个通用 LLM 才能学有限任务。Qwen3 官方披露约 36 万亿预训练 tokens、119 种语言/方言，说明通用基础能力有很大的数据与计算投入；这不是本项目有限候选任务的最低数据要求。[Qwen3 官方说明](https://qwenlm.github.io/blog/qwen3/)

三种路线需要分清：

| 路线 | 对本项目意味着什么 | 当前建议 |
| --- | --- | --- |
| 从零学习公开文本与语义标签 | 保留自有网络、训练循环和 tokenizer，增加有效覆盖与预算 | 继续作为学习主线和对照基线；不能只重复扩模板 |
| Qwen 作为离线教师 | 学生仍随机初始化，用教师的标签/候选分布补充监督 | 值得独立比较；先验收教师，再决定是否扩大蒸馏 |
| 直接加载 Qwen 权重并微调 | 采用 Qwen 的 decoder 架构、词表和权重体系 | 是另一条模型路线，不能把其权重直接塞入当前 encoder |

蒸馏不是复制一个“基础权重”。标准输出蒸馏让学生学习教师在训练输入上的分布；DistilBERT 的预训练蒸馏还涉及语言建模和表示对齐，不能把少量 yes/no 标签称为获得了通用基础语义。[知识蒸馏原论文](https://arxiv.org/abs/1503.02531)、[DistilBERT](https://arxiv.org/abs/1910.01108)

建议对原计划作以下调整，尚未新增训练或下载教师模型：

1. **保留一次有上限的优化排查。** 固定总更新预算与三种子，比较 `1 -> 2 -> 4 -> 8 -> full` 课程，检查旧家庭遗忘。先不同时换学习率、网络和监督目标。阶段学习率或证据监督作为之后的独立假设，不能无限叠加试到某个 seed 成功。
2. **自然语义覆盖成为下一轮主任务。** 审计已有公开 train 的实际独立源覆盖、语言/来源采样、token 数、否定/主体/条件/情态分项；保留受控绑定任务作为回归，不把合成任务全部达标设为开始自然语义研究的绝对前提。先利用已有数据，不以下载行数代替有效训练量。
3. **冻结新的隔离挑战集，再做路线选择。** 分组隔离人物/事实/改写/翻译，同源变体不跨 split；已多次查看的旧 test 与用户探针只作开发诊断。未明确事实、说话者信念和现实结果分别定义标签，不能把“我觉得会迟到”标成客观必然。
4. **若尝试教师，先评估其是否值得学。** 在独立开发集上测 Qwen 的中英否定、绑定、候选置换和信息不足；教师不是绝对真值。保留人工/可验证标签，记录教师错例。教师模型、量化、提示词、采样配置及数据版本固定并留指纹。
5. **做有限任务蒸馏对照，避免混淆来源。** A 为随机初始化 + 公开硬标签；B 使用相同学生、tokenizer、输入、种子和更新预算，在 A 基础上增加教师候选分布。额外合成/改写数据另做 C，并提供同样扩数据但无软标签的对照，区分“多数据”与“教师信号”。记录教师成本，不把学生更新数相同说成总成本相同。

```text
                          已有公开 train / 经审查的补充数据
                              |                    |
                              |            Qwen 冻结教师（离线）
                              |                    |
                              |             标签 / 候选分布文件
                              |                    |
随机初始化学生 A <--- 硬标签监督        硬标签 + 教师监督 ---> 随机初始化学生 B
                              \                    /
                               相同的独立评估与资源记录
                                        |
                         学生独立推理 -> 候选 softmax -> Ruby JSON
```

教师使用自己的 tokenizer，学生使用统一的自然语义 train tokenizer；对齐的是候选含义/顺序，不是 token ID。当前 400-token 关系 checkpoint 不可直接重解释成 12k 公共语义 checkpoint。教师与学生架构不同也可做候选输出蒸馏，但隐藏层/attention 蒸馏需要额外对齐设计。

若取得真实候选分数，可比较 `CE(gold, student) + lambda * T^2 * KL(teacher_T || student_T)`，其中 T 是蒸馏温度，区别于推理后校准温度。候选分布需要从明确的评分协议得到，例如验证过的单 token 选项代号、固定提示词、全候选打分与排列检查；自由生成“我有 90% 把握”不是 logits。只有最终答案时应称伪标签训练，不伪装成软概率蒸馏。当前代码尚未实现教师分布导入或该损失。

本机 6 GiB 下建议教师和学生分阶段运行：先用本地量化推理后端产出文件、退出教师释放资源，再用 Ruby/Torch.rb 训练学生；Ruby 可用 Faraday 调本地服务，不必把算法实现迁到 Python。具体 Qwen 型号、上下文长度、量化和显存适配尚需本机验证，不以 4bit 权重大小代替完整运行内存，也不承诺小学生能保留教师的全部能力。

当前推荐仍是**从零主线 + 有上限的优化排查 + 自然语义覆盖；教师蒸馏作为可比较的补充路线**。如果目标改为尽快获得可用效果，优先验证教师方案更合理；如果坚持完整从零学习，继续公开语料/监督也合理，但应按有限任务设目标，而不是期待本机短训重建通用 LLM。此处是策略建议，不代表用户已切换路线或教师实验已有结果。

用户进一步提出跨场景复用：建议把教师采集、产物和损失抽为 `EasyAI::Distillation`，Decision 只保留任务适配与训练接入。具体职责、目录、信号对齐与验收见[可复用蒸馏设计](distillation.md)，尚未实现。

## 8. 后续每轮都应保留的最小记录

- 假设、唯一主要变量、固定配置和预算；训练/验证/测试的源分组与指纹。
- 全部种子，包括拟合失败、未运行阶段与回退；选择依据是 validation，不能事后挑 test 最好的 seed。
- 更新数、样本行次、独立源覆盖、有效 token 数、时间与显存；“相同步数”不等于相同计算量。
- train 与 held-out、每语言/来源、成对与四条组全对率；再加状态/问题消融、候选重排和简单先验基线。
- 能支持的结论、不能支持的归因、下一条可证伪假设。旧 test 已反复用于讨论时明确标为探索性证据。

实际证据入口：`runs/decision/pipeline-showcase`、`expanded-matching`、`semantic-supervised-v2`、`semantic-mlm-v1`、`relations-v2-comparison`、`relations-v3-comparison` 和 `relations-v3-coverage-{1337,2027,3407}`。各阶段文档提供命令、配置和子目录位置；本文件不替代原始日志。

数据、tokenizer、权重和原始报告继续被 Git 忽略，交接需要另外复制完整产物；Git 保存本文、代码和精选图。原 [Learning 训练展示](../../README.md#learning-训练效果展示保留)保留，它属于独立教学实验，不能混入 Decision 成绩。

## 9. Gold-supervised exposure: a limited gain and a rejected augmentation

The completed [coverage experiment](coverage.md) fixes architecture, tokenizer and supervision, uses three seeds per data variant, preserves selections within 1k/4k budgets, and opens a newly reserved public-dev challenge after all training. It retains every seed. One expanded run recovered from GPU contention with smaller microbatches; that execution deviation is recorded rather than hidden.

| Intervention | Challenge source-macro accuracy | Mean raw NLL | Decision |
| --- | --- | --- | --- |
| Original data, 1k → 4k updates | 54.31% → 57.36%; all seeds improve | 0.85508 → 0.88801; all seeds worsen | Narrow pass on the predeclared accuracy gate; probability improvement not established |
| Original → wording-expanded data, both 4k | 57.36% → 55.39%; only one seed improves slightly | 0.88801 → 0.87430 | Fails the accuracy gate; do not adopt as an improvement |

Useful lessons:

- **Available rows are not learning exposure.** The 1k trajectories visit about 22% of unique rows, and 4k about 58%; source balancing revisits the small dataset much more often. Earlier validation-selected weights see less than those budget-end totals. Count retained row visits, source groups and tokens explicitly.
- **More expressions are not more facts.** The 23,280 wording variants retain the same 38,413 source groups, add about 8% token work, and reduce accuracy here. They also change within-source sampling frequencies, so this experiment cannot isolate wording from repetition/label proportions. This failure does not rule out adding independently useful gold examples.
- **Accuracy and probabilities need separate checks.** Longer training passes the accuracy rule by only 0.05 points above its threshold, while NLL, mean Brier and mean ECE worsen. Better argmax predictions do not establish trustworthy probabilities. Temperature calibration can adjust confidence, but cannot fix incorrect rankings or bindings.
- **An aggregate gain can coexist with weak conditioning.** Original/4k BoolQ accuracy remains below its training-majority diagnostic. OCNLI validation diagnostics are much more sensitive to shuffling the hypothesis than the premise. DuReader strongly uses the answer/state. Measure each task instead of describing all of them as general reasoning.
- **Known successes and failures both belong in the record.** Seed 2027 handles the four Chinese lateness probes but still fails English negation and person binding. Wording expansion makes the incorrect English answer more confident. These familiar probes are development diagnostics, not new held-out evidence.
- **Audit categories are not semantics.** Word-presence rules were removed from both accounting and selection after user review. Use declared metadata for exposure; use explicitly annotated phenomenon slices when needed. Unicode handling and metadata tests do not demonstrate understanding in Japanese, Korean, Arabic or any other untrained language.

The next hypothesis is that task-aligned, reviewed gold pairs can force dependence on the state for a fixed question/hypothesis. It remains to be tested with grouped splits and a new reserved evaluation, alongside probability-quality criteria. Neither another unbounded training extension nor automatic model growth follows from these results. The completed challenge is now observed; future tuning must not keep calling it fresh.

## 10. Evidence supervision and evaluation targets (2026-09-30)

This round adds a question-conditioned supporting-sentence head to the scratch-trained public model and compares answer CE with answer + 0.2 × evidence CE. Both arms reuse the same public parent for each seed, the same tokenizer, samples, replay and fixed budget. The implementation, corpus construction, results and reproduction live in [evidence.md](evidence.md).

Three lessons apply independently of the final score:

1. **A narrow acceptance threshold is not a definition of general semantics.** The original 95% per-language target concerns simple, fully specified facts. It remains an aspirational diagnostic; stable performance against meaningful baselines across natural domains takes priority. Published Jev/open-model percentages use different benchmarks and training settings and cannot justify directly raising or lowering this test's threshold.
2. **A larger generated test is not necessarily broad evaluation.** The 6,144 test rows come from only 32 actor/action families and fixed rendering templates. A familiar/novel-expression comparison isolates one form of shift but does not establish intent routing or emotion/news understanding. A supplemental 3,793-row zero-shot panel was therefore frozen before final evaluation, with full label sets, training-text overlap checks, majority/chance baselines and all-seed reporting.
3. **Auxiliary learning and answer improvement are distinct.** On the 128-row capacity check, supporting-sentence accuracy improved from 51.56% to 75.78%, while answer accuracy changed from 75.00% to 74.22%. Learning where to look is not yet proof of using that information correctly. Keep both measurements rather than present only the improved auxiliary metric.

A concrete engineering failure was also fixed: retaining zero gradients for an intermittently unused head advances AdamW state differently after checkpoint resume. Clear absent gradients consistently, and test interrupted versus uninterrupted training with mixed annotation coverage. Torch.rb 0.23's Parameter setter cannot clear a nil gradient safely; the Tensor setter can, and the local implementation is covered by regression tests.

Final main and generalization results are recorded in the linked experiment document; the original gates and all seeds are retained, including negative outcomes.


### Review completion — 2026-10-01

The shared evaluator now rejects malformed probability vectors and derives correctness from the distribution rather than trusting a saved flag. Replaying all nine saved prediction files leaves the reported results unchanged. Treat validity checks as part of the evaluator contract, including failed inputs in the denominator. Pin downloads and verify cached files too; a revision-unpinned dataset service can change bytes even when its URL stays the same.

The answer/evidence curves now accompany the README. Training answer loss falls while validation NLL rises; every main arm selects step 100. This warns against treating a decreasing training curve or a stronger auxiliary metric as successful transfer. The [next protocol](next-experiment.md) separates small-set fitting, natural-task supervision and context coverage, so each proposed remedy has a measurable failure mode.


### Fitting and positional comparison — 2026-10-01

The [four-condition fitting round](fitting.md) separates starting representation from positional encoding. At the same additional 2,000-update budget, our own parent fits all 128 rows under sinusoidal positions and RoPE; RoPE reaches the observed gate at 1,000 updates versus 1,800. Random starts finish at 75% and 87.5%, respectively. RoPE's random-start gain is confined to fully fitting Chinese here; English remains at 75%. This is one seed on a training set, not natural-language generalization. The parent also has more lifetime training compute.

The failed random/sinusoidal fit has 32 errors, all mixed-truth states and evenly split across languages. Random/RoPE retains 16 English mixed-truth errors. Keep these failures: they identify a narrower unresolved binding/optimization problem than “the model understands nothing.” Both fully fitted parents drop near chance when states are shuffled, while confidence remains high. State dependence and training fit do not certify calibrated reasoning.

Instrumentation caught an invalid first harness attempt: an unconditional same-device Torch.rb `Module#to` replaced Parameters after optimizer creation. Scores stayed identical while gradients accumulated on the live model. Preserving parameter references and testing a real subsequent update fixed it; the attempt is retained and excluded from comparison. Earlier evidence training's internal validation did not use this evaluator during updates. A noisy batch loss alone would have hidden the problem.

Carry forward the own-parent/RoPE candidate into independent task/relationship evaluation with a sinusoidal control. Do not grow the model merely to remedy the old 300-update fitting result, and do not mistake success on 128 rows for broad semantics. Cold-start learning-rate/sampling questions remain open.


### Focused v0.1 delivery and bounded correction — 2026-10-01

See [v0.1 results and reviewed loss charts](v01.md). The first request-domain fit improved substantially over its own broad initializer, but decreasing training loss did not ensure held-out improvement. Validation NLL selected update 1,200; later updates overfit. Its calibration policy targeting 90% selected accuracy did not maintain that point target on independent tests. Calibration performance is an estimate, not a guarantee.

The bounded correction combined natural within-language sampling, a smaller learning rate and stronger calibration guards. Fresh independent selected accuracy reached 90.32% English / 91.64% Chinese at 77.5% / 80.75% coverage, while full accuracy remained 78.25% / 81.0%. Multiple factors changed, so this does not isolate any one factor's contribution. Acceptance panels differ, prohibiting a paired claim about raw accuracy changes. The fresh corrective panel came from unused official TRAIN groups reserved before fitting, not from reopening the first test. Both panels are now observed regression material.

The user approved a usable scoring preview and clarified the 80% direction is advisory. Keep all historical metrics, including failures, and distinguish delivered capability from broader aspirations. This is 18-domain bilingual scoring, not established general reasoning. Weak class recall and unseen option semantics remain concrete next targets. The artifact carries preview status, exact checkpoint identity and CPU/CUDA/runtime verification, while the strict publisher retains its original checks. No model weights or evaluation labels changed to obtain preview status.
