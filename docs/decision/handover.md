# Decision 开发交接：人物绑定与下一轮优化

更新：2026-09-28。代码基线：`d57b118`（`Feat: add Decision training and inference pipeline`）。本文记录该提交之后的用户复测、实际排查与下一轮计划；下文标为“待做”的能力尚未实现。

**当前结论：训练与推理流程可运行，课程学习带来改善，但人物—事实绑定仍不稳定，而且会错在已有训练样本上。下一轮先修正训练对照和课程设计，不直接扩大模型或宣称通用语义已经可用。**

## 1. 接手时的约束与状态

- 保持 Ruby / Torch.rb 路线：Ruby 实现算法和训练流程，LibTorch/CUDA 执行张量计算；当前选择从零训练，不导入外部基础权重或教师输出。
- 本机 Ruby 3.4.5、Torch.rb 0.23.0，Quadro RTX 3000、6 GiB。GPU 优先，进程显存软预算 4096 MiB；容量不足时缩小批次/分块，再回退 CPU。当前实现 FP32。
- 第一项产品能力是输入 `state / question / options`，输出候选概率。候选 ID 可为数字或字符串，输出键统一为字符串。JSON 由 Ruby 组装，网络不生成 JSON 文本。
- 语言目标不局限英语；本轮关系与公开语义实验是中文、英文，不能因此声称已有任意语言能力。
- 正式能力在 `lib/easy_ai/`；历史教学代码在 `learning/`；实验编排在 `benchmarks/decision/`。配置每行一个参数，README 与旧 learning 曲线继续保留。
- `data/`、`runs/`、下载文件、权重被 Git 忽略。文档和精选图表在 Git 中。只 clone 仓库**不会得到本机训练产物**。
- 最近代码验收：正式测试 71 tests / 640 assertions，教学 7 tests / 14 assertions，RuboCop 95 files，全通过。这是现有代码的验收结果，不是未来修改自动通过的保证。

已完成：数据准备、MLM/候选训练、checkpoint/续训、校准、CPU/CUDA 推理、RoPE、两种池化和联合编码对照、成对评估、课程实验、训练图表与显存累积修复。**当前所有课程种子仍未通过泛化门槛。**

## 2. 正在测试的权重与可复现入口

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

数据缺失时可重建（输出目录必须不存在；不要覆盖现有 v2）：

```bash
bundle exec ruby bin/easy-ai prepare-relations --output data/decision/relations-v2 --vocab-size 400 --seed 1337
```

一条命令重训当前课程基线，自动生成新 run 目录、日志与图表：

```bash
bundle exec ruby benchmarks/decision/relations.rb --variants rotary --seeds 1337,2027,3407 --curriculum --patience 0 --steps 2000
```

每个 seed 是 1000+2000 步，第二阶段重置优化器和 warmup。新机器重训后的 checkpoint 目录带新后缀，以新 run 的 `summary.json` 为准，不会重建上述同名路径或承诺跨设备逐 bit 一致。没有 `--curriculum` 时，完整阶段重新随机初始化。

单独评估已有最佳权重：

```bash
bundle exec ruby bin/easy-ai evaluate-relations --checkpoint runs/decision/relations-v2-curriculum/choice/best --data data/decision/relations-v2/validation.jsonl --batch-size 128 --controls
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

代码事实：`RelationCorpus` 把 `question_flip` 写入 `contrast_group`，`PairSampler` 每次将这两个样本一起采样。目前没有专门的主体切换/人物角色互换训练分组；现有评估包含 question_flip、fact_flip、irrelevant_fact、order 四类。

待验证假设：现有配对更容易强化问题肯定/否定变化，而对“否定属于谁”的约束不足。全局池化可能弱化局部关系，但已有 cross-attention，不能把错误直接归因于“没有 attention”。单家庭热身、固定学习率、有限句式都可能影响训练；需要逐项对照。

## 5. 下一轮待办与顺序（尚未实现）

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

接手第一步：核实权重与数据版本，重现本页买票成对样本的判断错误；随后实现 P0 的主体/角色评估，再做 P1 的采样对照。这里记录了可执行顺序，不表示这些新功能已经完成。
