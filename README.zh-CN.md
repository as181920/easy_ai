# easy_ai

**语言：** [English](README.md) | 简体中文

**在线学习课程：[Easy AI Learning](https://easy-ai-learning.code-li.com/)** — 从基础神经网络到经典模型架构、训练方法与实践，按课程顺序逐步学习。

用 Ruby 学习并实现神经网络，通过循序渐进的课程和可运行的模型实验掌握原理。算法、训练循环、分词器和优化器可在仓库中阅读；张量计算与自动微分交给 Torch.rb / LibTorch，可使用 CUDA 或 CPU。

## Learning 学习课程

从[课程总览](learning/README.md)开始，也可以直接进入[在线学习站](https://easy-ai-learning.code-li.com/)，使用搜索、章节导航和源码链接。00–16 章从可手算的小例子逐步推进到完整的 Ruby/Torch.rb 实验，后续课程复用前期实现，形成连续学习路径。

| 阶段 | 章节与主题 |
| --- | --- |
| 基础与训练 | [00：数学与数据](learning/00_foundations/README.md)、[01：基础神经网络](learning/01_basic_nn/README.md)、[02：SGD、AdamW 与正则化](learning/02_training/README.md)、[03：权重、梯度与诊断](learning/03_diagnostics/README.md) |
| 表征与视觉 | [04：自编码器](learning/04_autoencoder/README.md)、[05：CNN](learning/05_cnn/README.md)、[06：ResNet](learning/06_resnet/README.md) |
| 序列学习 | [07：分词器](learning/07_tokenizers/README.md)、[08：RNN/LSTM/GRU](learning/08_rnn/README.md)、[09：Seq2Seq](learning/09_seq2seq/README.md) |
| 注意力与语言 | [10：注意力与掩码](learning/10_attention/README.md)、[11：Transformer](learning/11_transformer/README.md)、[12：GPT](learning/12_gpt/README.md) |
| 生成、迁移与交互 | [13：VAE/GAN/Diffusion](learning/13_generative/README.md)、[14：迁移学习与 LoRA](learning/14_transfer_learning/README.md)、[15：强化学习](learning/15_rl/README.md) |
| 完整实验 | [16：综合实践](learning/16_capstone/README.md) |

先完成 00–03，再选择视觉或语言分支，之后学习生成、迁移或强化学习。每章包含文档、数据准备、训练、推理和实测结果。核心测试验证公式、梯度、掩码和参数更新，不依赖训练是否收敛。

- [本地阅读指南](docs/learning-site.md)：运行 Ruby 文档站，使用跨章节导航与搜索。
- [全课程精简打印版](docs/learning-course-print.md)：核心概念、公式与必要代码。
- [课程实现](learning/lib/easy_ai_learning/)：各章复用的教学组件。

### 环境准备

Ruby 3.4，Bundler，以及可用的 LibTorch 环境：

```bash
bundle install
bundle exec rake test:learning
bundle exec rake lint
```

本机已验证 Torch.rb 0.23.0 和 Quadro RTX 3000 6 GiB；安装 LibTorch 时需匹配 Torch.rb 的兼容版本及 CPU/CUDA 构建，参见 [Torch.rb 安装说明](https://github.com/ankane/torch-rb#installation)。仅安装 gem 不代表 CUDA 已可用。

### 基础神经网络示例

最基础的神经网络训练示例（不需要语料和 GPU）：

```bash
bundle exec ruby learning/01_basic_nn/logic.rb
bundle exec ruby learning/01_basic_nn/train.rb
```

训练一个只计算 XOR 的神经网络，打印训练进度、学到的参数和 unicode_plot 函数图；输出保存到被忽略的 `runs/learning/basic_nn/logic-gates/`。默认 seed 1337 的 CUDA 训练运行 436 步后，四个 XOR 输入的阈值判断全部正确。详见 [基础 NN 学习说明](learning/01_basic_nn/README.md)。

基础 NN 默认使用 Torch.rb / CUDA（不可用时回退 CPU）；[README 函数图](learning/01_basic_nn/README.md#observed-runs-and-plots)分两组展示：固定 AND/OR/NAND/XOR 逻辑函数；模型的 ReLU、训练 loss、score 热力图和 3D 曲面、XOR 预测边界。终端训练报告也保留 loss 曲线。

### 运行全课程

```bash
bundle exec ruby learning/run_all.rb
bundle exec rake test:learning
```

运行器按章节顺序准备数据并执行实验。神经网络训练优先使用 CUDA，不可用时回退 CPU；`--device cpu` 可显式选择 CPU。产物保存到被忽略的 `runs/learning/` 目录。逐章命令与实验设置见[课程总览](learning/README.md#an-executable-learning-path)。

## 已实现模型：Decision

**EasyAI::Decision** 给定状态、问题和多个候选，直接输出候选概率。下面介绍其架构、训练流程和实测结果。

```bash
bundle exec ruby bin/easy-ai --help
bundle exec rake test
```

Decision v0.1 以**仅含模型的评分预览版**交付，面向中文和英文请求领域：`runs/decision/v0.1-preview`。使用 `EasyAI::Decision::Release.load("runs/decision/v0.1-preview", device: "auto")` 加载。修正后的模型总体准确率为**英文 78.25% / 中文 81.0%**；高置信度判断的准确率为 **90.32% / 91.64%**，覆盖率为 **77.5% / 80.75%**。80% 目标仅作为后续优化参考；原先严格验收失败的记录仍然保留。参见[结果、图表、推理与训练命令](docs/decision/v01.md)。通用是非推理尚未验证，当前不包含业务集成。

已完成的 [Decision v0.2 事实评分实验](docs/decision/v02.md)使用 27,520 条双语训练数据、相同预算的交叉熵（CE）与间隔损失对照，以及 7,144 条验收判断。CE 将事实判断按来源和语言的宏平均准确率从 **46.33% 提升至 67.33%**；间隔损失达到 **66.47%**。CE 的整对判断全对率为**英文 64.53% / 中文 99.92%**，但英文主体绑定、中文缺失信息处理和路由能力保持仍然失败。**v0.1 仍为交付预览版**；v0.2 权重仅用于诊断。文档包含加载路径、完整训练图、数据接触与来源审计，以及后续优先事项。

已完成的 [Decision v0.3 实验](docs/decision/v03.md)采用路径 A，基于我们自训的 v0.1 权重，修正主体、查询与顺序监督，并显式加入未知信息样本。两组均完成 2,000 次 CUDA 更新。在同一组新评测上，候选模型将事实宏平均准确率从 **41.63% 提升至 52.23%**，自然 QA/NLI 从**英文 48.50% / 中文 48.00% 提升至 64.50% / 56.75%**；但已知事实判断下降至 **15.43% / 16.88%**，混合真假绑定仍为 **0.72% / 0%**。事实逐条准确率下降，汇总指标的提升不代表有效的事实推理。**v0.3 未晋升为交付模型，仍交付 v0.1。** 报告包含已审查的训练图、诊断加载路径、已核验的数据接触、运行检查和失败经验。[已执行计划](docs/decision/v03-plan.md)也可查阅。

[已完成的 v0.4 实现](docs/decision/v04.md)增加了经审查的监督数据、完整反事实样本族、分阶段 CE 训练器和类别/分组检查。独立的 128 条样本拟合停在 87.5%：不同事件样本通过，但混合主体样本即使在训练集上也仅为 50%。全部 1,000 次更新使用 CUDA；运行与内存检查通过。按照停止规则，试验性训练和验收尚未运行，v0.1 仍为交付版本。这个训练成绩不代表泛化结果。参见[计划与流程](docs/decision/v04-plan.md)。

[v0.5 开发计划](docs/decision/v05-plan.md)将在同一组经审查的核心数据上比较分离式与联合式关系编码，验证位置编码和权重迁移，并以可重复的主体绑定学习结果作为监督泛化、未知信息训练和校准的前置门槛。目前仍处于计划阶段，实现与训练尚未开始。

`Release.load` 和 `Predictor.load` 的 `device:` 均接受 `:auto`、`:cpu`、`:cuda` 及对应字符串。
`Release#probabilities(state:, question:, options:, language: nil)` 可以不传语言参数，直接接受多语言文本。各语言共用一个分词器和一个 checkpoint；`language:` 仅选择已评估的路由置信度策略。`route` 的预定义候选描述仍需要指定语言。

```text
state --------------------> shared bidirectional encoder ----> state memory
question + each option ---> shared bidirectional encoder ----> cross-attention
                                                               |
                                                   masked mean + scalar score
                                                               |
                                                   softmax(scores / temperature)
                                                               |
                                                   Ruby Hash -> JSON probabilities
```

语义判断来自训练后的网络，不使用关键词规则。覆盖审计仅依据数据集中声明的元数据。分词器与推理接口接受多语言文本，但当前中英实验尚不能证明其他语言的语义能力。参见[覆盖实验与限制](docs/decision/coverage.md)。

默认 `small` 配置的参数层级（共享模块只计数一次）：

```text
ChoiceModel                                    11,484,929 params
|-- Shared Encoder                             10,826,240
|   |-- Token Embedding [32000, 256]             8,192,000
|   |-- Sinusoidal Positions + Dropout                  0
|   |-- Encoder Block x 4                       2,633,728
|   |   |-- LayerNorm x 2                           1,024 / block
|   |   |-- Self-Attention (4 heads x 64)          263,168 / block
|   |   `-- FFN (256 -> 768 -> 256)                394,240 / block
|   `-- Final LayerNorm                               512
|-- Interaction Block x 1                         658,432
|   |-- LayerNorm x 2                               1,024
|   |-- Cross-Attention (4 heads x 64)            263,168
|   `-- FFN (256 -> 768 -> 256)                    394,240
|-- Masked Mean Pooling                                 0
|-- Shared Score Linear (256 -> 1)                     257
`-- Temperature + Softmax                              0 network params
```

温度是独立校准得到的一个标量。MLM 输出使用 embedding 转置，不另建词表投影参数。`smoke` 使用 vocab 4096、hidden 32、encoder 2 层、FFN 64，共 156,801 参数，仅用于快速跑通流程。

改进后的 `massive.yml` 使用同一共享 encoder，增加输入 LayerNorm（512 参数）和显式状态匹配投影（262,400 参数），合计 **11,747,841 参数**：

```text
state -> encoder -> memory ---- masked mean -> s ---------+
                                                        |
question + option -> encoder -> cross-attention -> q ----+
                                                        |
                              [q, s, q*s, abs(q-s)] (1024)
                                                        |
                                   Linear(1024,256) + GELU
                                                        |
                                     Linear(256,1) -> softmax
```

旧权重继续使用原结构；新配置需要重新训练，结构变化不会靠加载旧文件自动生效。

目前完成了从公开数据准备、自训分词器、随机初始化 MLM 预训练、候选监督训练，到续训、校准、评估和推理的完整流程。已有本机 GPU 验收产物，但短程训练产物仅用于验证工程流程，不能当作具有通用判断能力的预训练模型。

```text
easy_ai/
|-- lib/easy_ai/
|   |-- decision/           # 正式维护的候选概率能力、数据、训练、推理、扩容
|   |-- distillation/       # 可复用的教师采集、产物与监督损失
|   |-- nn/                 # 注意力、FFN、编码器块
|   |-- optim/              # 可保存状态的 Ruby AdamW
|   |-- runtime/            # GPU 优先、显存预算、CPU 回退
|   `-- tokenizers/         # 自写 byte BPE / Rust gem 后端
|-- learning/               # EasyAILearning: 基础 → 训练/诊断 → AE/CNN/ResNet → 序列/GPT → 生成/迁移/RL
|-- test/                   # 正式库测试；learning/test 单独运行
|-- config/decision/        # small 正式起点 / smoke 流程验收
|-- examples/decision/      # 公共 API 示例、JSON 请求
|-- benchmarks/decision/    # 分词、实际训练、GPU 验证
|-- docs/decision/          # 架构、使用、验收记录
|-- data/                   # 下载、数据集、分词器（gitignored）
|   |-- learning/           # 原有 TXT 学习语料，教学训练默认目录
|   `-- decision/           # 候选概率模型的数据与分词器
`-- runs/                   # 权重、优化器、日志、校准结果（gitignored）
```

### 一条命令完成 Decision 训练

当前按学习路线推进**从零训练**：新语义实验已准备 107094 条中英监督数据，模型 6,627,841 参数，不导入外部基础权重或教师输出。一条命令包含自训 MLM、候选监督、输入扰动诊断、校准和效果图：

```bash
bundle exec ruby bin/easy-ai semantic-pipeline
```

默认 MLM 1000 步 + 监督 1000 步；`--mlm-steps 0` 可运行从随机初始化直接监督的对照。已有本地数据会复用，缺失时才通过 Faraday 下载。每次输出到独立 `runs/decision/` 目录，有进度、日志、HTML/PNG/SVG 图表。公开数据的任务定义、许可、拆分、模型 ASCII 图与实测见[从零语义训练](docs/decision/semantics.md)。目前仍属学习实验，不能把这批短程权重视为已经掌握通用语义。

长文本验证曾发生显存持续累积，已通过限制临时张量生命周期与明确回收修复。候选/MLM 各连续 5 轮验证实测预热后显存稳定在 1178 MiB、无 CPU 回退，复现见[内存与显存回归](docs/decision/memory.md)。

本轮从零 MLM + 监督的实测曲线如下。相同 test 上，来源宏平均 accuracy 从直接监督的 57.56% 变成 55.11%；这个预算下 MLM 没有改善下游判断，迟到/否定探针仍有错误，详细对照和后续排查见上述语义训练说明。

![从零 MLM 与候选监督曲线](docs/images/decision-semantic-loss.png)

最近的受控实验（2026-09-29）：将人工标注监督训练从 1,000 次延长至 4,000 次更新，三种子新挑战集的来源宏平均准确率从 **54.31% 提升至 57.36%**，但原始 NLL 变差。增加任务和候选措辞变体后，准确率降至 **55.39%**，因此不接受为改进。英文否定和人物绑定仍未通过已有探针。参见[完整结果、训练曲线、权重路径与下一轮实验](docs/decision/coverage.md)；这些是实验权重，不是可靠的通用语义模型。

![人工标注监督的覆盖对照](docs/images/decision-coverage-comparison.png)

2026-09-30 结果：证据监督未明显改善新表达准确率（50.71% 对 50.76%）或绑定能力（均为 7.60%）。更广泛的自然数据和公共基准评估还暴露出迁移能力不足与输入长度限制问题。95% 目标是局部诊断指标，不是整体验收标准。基准工具与可复现下载现已审查并测试。参见[交接记录](docs/decision/handover.md)与[下一轮实验协议](docs/decision/next-experiment.md)。

![证据监督对照](docs/images/decision-evidence-comparison.png)

![三种子的答案与证据损失](docs/images/decision-evidence-loss.png)

已完成的[拟合诊断](docs/decision/fitting.md)在 128 条双语训练样本上比较正弦位置编码和 RoPE。我们自训的父模型在两种编码下均达到 100%（RoPE 在第 1,000 次更新达到门槛，正弦编码在第 1,800 次）；随机初始化最终为 75% / 87.5%。这些是单种子的训练拟合结果，不代表泛化。没有任何预测模型晋升为交付版本。

![位置编码与初始化的拟合对照](docs/images/decision-fitting-comparison.png)

已完成的[自然任务试验](docs/decision/natural.md)增加了全候选 AG News 和双语 MASSIVE 监督。在同一组新评测上，主要来源宏平均准确率从自训起始父模型的 **42.43% 提升至 61.04% / 60.84%**（正弦编码 / RoPE）。提升集中在已训练的新闻和领域任务；留出的 Emotion 仅为 **6–7%**，绑定组全对率为 **4.6–5.7%**。这说明学到了具体任务，不代表通用语义能力，也不能证明 RoPE 更优。没有任何预测模型晋升为交付版本。文档记录了一键复现、实验加载路径、内存与数据接触审计、失败分布和下一项诊断。

![自然任务试验对照](docs/images/decision-natural-comparison.png)

![自然任务训练与验证曲线](docs/images/decision-natural-loss.png)

[证据监督实验](docs/decision/evidence.md)使用我们从零训练的父模型，对比答案 CE 与答案加支持句 CE。实验从 32 个独立样本族中留出 6,144 条受控中英测试样本，另有一组新的 809 条公开数据测试。迟到探针是非正式检查，不是验收标准。文档包含模型 ASCII 图、一键复现和公开的 Jev / 开放模型对照；95% 目标是局部任务可靠性门槛，不是通用语义理解的定义。

针对否定错误，新增了中英关系学习对照：先验证 64 条样本能否完全拟合，再训练 13824 条可验证标签的关系数据，按人物/动作家庭与句式隔离评估。相同 663 万参数、1000 步预算下，v2 小样本实验的原位置编码 accuracy 为 75%，RoPE 为 100%；这是训练拟合结果，泛化需要另行评估。

完整数据进一步对照后，直接训练的三种子新句式 test 平均约 69.90%；先学小样本再训练完整数据的课程版约 80.03%，各次为 78.91%–80.78%。单独延长到 6000 步没有改善该种子的最佳验证结果。课程版有进展，但尚未达到每语言 95% 与成对反例门槛，仍不能视为通用语义模型。

```bash
bundle exec ruby benchmarks/decision/relations.rb --variants all,candidate,rotary --seeds 1337,2027,3407
```

这条命令包含进度、门槛检查、独立泛化训练与每轮 HTML/PNG/SVG 曲线；`--sanity-only` 可只做小样本排查。模型层级、数据拆分、成对反例及后续 scaling 策略见[关系学习实验](docs/decision/relations.md)。

复现本轮课程学习结果（仍然全部从零开始，不用外部权重）：

```bash
bundle exec ruby benchmarks/decision/relations.rb --variants rotary --seeds 1337,2027,3407 --curriculum --patience 0 --steps 2000
```

每个种子先训练 1000 步 sanity，再用这组自训权重训练完整数据 2000 步；总计 3000 步。报告同时列出完整训练集、验证集与两种测试句式的成绩，保留失败门槛，不自动推广为默认模型。

最新用户复测确认：两条“买票”例子已在训练集中，旧课程模型仍不能稳定判断谁买了票。现已完成主体/角色评估、四条成组采样及三种子对照：平均 test accuracy 从 **80.03% 到 82.55%**，混合真假四条组全对率从 **33.96% 到 48.33%**，但部分种子退步，仍未达标。扩大为 512 条代表性热身的后续试验也不稳定，完整结果、失败记录与可加载权重见[人物绑定对照](docs/decision/binding.md)。新关系训练与评估统一使用 v3 元数据，历史 v2 权重可在文本和 tokenizer 不变的 v3 数据上补测。接手入口为[开发交接记录](docs/decision/handover.md)。

![主体绑定的全部种子对照](docs/images/decision-binding-comparison.png)

原有 MASSIVE 多语言意图匹配实验保留：

```bash
bundle exec ruby bin/easy-ai pipeline --config config/decision/massive.yml --train-limit 2000 --backend native --vocab-size 8000 --mlm-steps 0 --choice-steps 1000 --eval-every 100
```

从随机权重开始监督训练，自训 native BPE；五语言共 10000 条训练行 / 2000 个源分组，其他 split 各保留 200 个源分组。
包含标签平衡、动态负候选、最佳 validation 权重选择、状态消融、独立校准与测试、进度和图表。
本机完整运行约 3.5 分钟。该任务训练意图选择，**还没有获得通用中文问答或否定语义判断能力**。

要学习包括 MLM 的原始小数据流程，仍可运行：

```bash
bundle exec ruby bin/easy-ai pipeline
```

自动执行：本地 MASSIVE 数据准备 → 自训 Ruby BPE → MLM 预训练 → 候选监督训练 → 独立温度校准 → test 评估 → 绘图。
默认使用 `small` 的约 1148 万参数模型、五种语言、每个 split 每种语言最多 200 条、每例 8 个候选，MLM 100 步、候选训练 300 步。
本地已有原始压缩包会直接复用；缺失时才下载。默认 GPU 优先，训练时显示 step、loss、最近验证 loss、设备及预计剩余时间。

每次创建独立的 `runs/decision/<时间>-<随机后缀>/`，控制台会打印路径；也可通过 `--output` 指定一个尚不存在的目录。
绘图使用本机已安装的 **gnuplot**，不依赖 Python；在新机器上需要先安装 gnuplot，缺失时会在训练前提示。

```text
runs/decision/<run>/
|-- pipeline.log              # 阶段进度、训练进度、错误信息
|-- train.log                 # 每次更新与验证的日志
|-- mlm/                      # 预训练 checkpoint、training.jsonl、metrics.jsonl
|-- choice/                   # 候选训练 checkpoint 与曲线原始数据
|-- calibrated/               # 校准后的推理 checkpoint
|-- stage-results/            # 校准与测试的完整 JSON 结果
`-- report/
    |-- index.html            # 浏览器打开：曲线、指标、按语言评估
    |-- loss.png / loss.svg   # MLM 与候选训练/验证 loss
    |-- evaluation.png / .svg # 校准指标与 test 可靠性图
    |-- memory.png / .svg     # 有 GPU 测量时：验证前后显存
    `-- loss.txt              # 终端 Unicode 曲线
```

训练 loss 是随机采样 batch 的交叉熵，验证 loss 来自独立数据；曲线没有平滑。它展示优化过程，不是梯度数值图，也不等于通用模型质量认证。
小数据实验应同时看验证曲线及 held-out test，不能只追求训练曲线下降。

更多选项、日志跟踪、报告重绘与中断恢复见[一键训练指南](docs/decision/usage.md#一键训练与效果图)。

- [使用指南](docs/decision/usage.md)：可复制的完整训练与推理命令、数据格式。
- [算法与设计](docs/decision/architecture.md)：模型选择、概率含义、缓存、扩层、资源策略。
- [本机验收](docs/decision/validation.md)：实际参数、时延、显存、测试与质量限制。
- [从零语义训练](docs/decision/semantics.md)：中英公开判断数据、MLM 对照、按任务诊断。
- [关系学习实验](docs/decision/relations.md)：否定绑定、位置编码对照、小样本拟合与独立泛化。
- [人物绑定对照](docs/decision/binding.md)：v3 数据、主体/角色评估、四条成组采样与实验入口。
- [迭代回顾与经验](docs/decision/retrospective.md)：从初版到 v3 的成功、失败、证据边界，以及从零训练与教师蒸馏的下一步取舍。
- [可复用蒸馏](docs/distillation/README.md)：离线教师内容、伪标签导出、候选级监督与 Decision 集成，包含原始设计链接。
- [首轮 Qwen 教师筛选](docs/distillation/teacher-v1.md)：可在 6 GiB GPU 上运行，但未通过候选顺序稳定性门槛；没有学生权重晋升为交付版本。
- [任务专用教师对照](docs/distillation/teacher-task-v1.md)：显式任务定义使顺序一致率下降（91.67% → 85.42%）；包含可复现对照、双方均正确的指标及失败经验。
- [人工标注监督覆盖实验](docs/decision/coverage.md)：完成六次运行对照；延长训练提升 3.05 个准确率百分点，但原始概率指标变差，措辞扩充则下降 1.97 个百分点。包含数据接触、内存回收、失败探针与复现；未使用关键词语义规则或预训练大语言模型。
- [开发交接记录](docs/decision/handover.md)：当前权重、用户复测证据、下一轮任务与验收标准；接手入口。
- [内存与显存](docs/decision/memory.md)：长文本验证的资源管理修复与连续测量。

配置遵循每行一个参数、优先人类可读性的约定。风格参考 easy_biz 的 RuboCop 习惯，项目继续使用 Ruby 库的目录结构。下载档案、训练数据、tokenizer 文件、权重和缓存请放入上述忽略目录；`examples/` 和测试代码可以正常提交。

### Decision 本机训练效果

改进配置的实际结果（2026-09-28，八候选、五语言，test 各 200 条；与旧实验使用同一组验证、校准和测试数据）：

| 指标 | 原始小数据实验 | 改进配置 |
| --- | ---: | ---: |
| 测试准确率 | 25.5% | **75.6%** |
| 测试负对数似然（NLL） | 1.95793 | **0.73891** |
| 中文 validation 原始状态 accuracy | 29.5% | **76.0%** |
| 中文 validation 打乱状态 accuracy | 29.5% | **21.5%** |

新实验使用更多训练数据、不同打分头与训练设置，不是单因素对照；这是采样八候选的结果，不是全 60 类 MASSIVE 官方成绩。
最佳验证权重在第 700 步，后续仍有过拟合，报告保留全部 1000 步曲线。详情见[诊断记录](docs/decision/diagnostics.md)。

![改进后的候选训练与验证曲线](docs/images/decision-matching-loss.png)

完整报告：`runs/decision/expanded-matching/report/index.html`；可加载的本地权重：

```ruby
predictor = EasyAI::Decision::Predictor.load("runs/decision/expanded-matching/calibrated")
```

候选 ID 支持整数、有限浮点数和字符串，输出键统一为字符串；`1` 与 `"1"` 同时出现会报重复。
实际手工检查中，“明天七点叫我起床”选中 alarm_set，“播放音乐”选中 music_play；但“来不及了要迟到了 / 会迟到么 / 会、不会”仍选错。
这些例子未用于训练。当前改进证明了意图匹配中的状态依赖，不能宣称已解决一般否定语义。

GPU 吞吐也已调整：有效 batch 同为 32，`4×8` 累积改成 `32×1` 后，短程实测从约 84 增至 281 条/秒（3.35 倍）。
完整运行中一次采样的训练进程显存约 1.85 GiB；这不是峰值承诺。输入更长时会触发原有缩 batch / CPU 回退策略。

以下保留原始实验，便于对照。其约 1148 万参数，五语言，每个 split 1000 行（200 个原始分组），8 候选，MLM 100 步、候选训练 300 步，完整流程约 5 分钟。

![Decision 的 MLM 与候选训练、验证 loss 曲线](docs/images/decision-loss.png)

MLM 的训练与验证 loss 都下降；候选训练后半段出现训练 loss 下降而验证 loss 波动上升，提示小数据过拟合。
本次最终 checkpoint 的 test accuracy 为 25.5%（采样八候选），NLL 为 1.95793；仅作为本地小规模实验记录，不代表通用判断能力。

后续状态依赖性检查发现：200 条中文 validation 中，打乱 state 或将其统一替换成“无上下文”，所有最终选择均保持不变。
这组旧权重尚未证明学会状态与候选的匹配；在这组 validation 上甚至未超过只看训练标签频率的基线。详见[候选学习诊断](docs/decision/diagnostics.md)。

![Decision 温度校准指标与 test 可靠性](docs/images/decision-evaluation.png)

本次温度拟合降低 calibration NLL，但 Brier 和 ECE 没有同时改善。报告保留原始曲线与指标，不筛选“好看”的点。
本机完整报告位于 `runs/decision/pipeline-showcase/report/index.html`；其他机器可执行一键命令生成自己的报告。
原始权重、语料和日志不随 Git 分发，README 的展示图单独保留在 `docs/images/`。
