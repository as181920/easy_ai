# 可复用蒸馏能力设计（尚未实现）

本设计回应“蒸馏能否独立于 Decision，供以后其他场景复用”。建议命名为 **`EasyAI::Distillation`**，作为与 `Decision` 并列的能力。首个使用者是候选概率任务；公开硬标签从零训练继续作为对照。本文不是已可运行 API 的说明，也不代表已下载、验收或训练 Qwen 教师。

## 职责与数据流

```text
                 input records + split/source identity
                                  |
                       Task adapter (e.g. Decision)
                       prompt / candidate mapping
                                  |
EasyAI::Distillation     Teacher backend (Qwen / others)
                        capability checks / retries
                                  |
                        immutable teacher artifacts
                        provenance / cache / validation
                                  |
                       Task adapter: align supervision
                                  |
                 student trainer + reusable loss primitives
                                  |
                   existing task evaluation / calibration
```

公共层负责教师后端协议、能力检查、离线采集、缓存/断点、版本记录、产物验证与通用损失。它不知道 `state`、`question`、“成立/不成立”或业务选项的含义。Qwen 是一种教师来源，不是蒸馏框架的固定依赖。

Decision 适配层负责将 `state/question/options` 转成教师请求、按候选映射监督、检查排列偏差与语义标签，并把监督送入已有 Trainer。推理仍由 Decision 的神经网络评分，Ruby 组装 JSON；教师只出现在离线准备阶段。

通用层不能假定所有任务都有相同损失或 tokenizer：

| 未来使用者 | 可复用内容 | 仍由场景实现 |
| --- | --- | --- |
| Decision / 分类 | 教师调用、候选分布存储、温度 KL、溯源与缓存 | 候选含义/顺序对齐、有效项 mask、硬标签 CE 与评估 |
| 向量表示 / 检索 | 后端与产物流程、适用的表示损失 | 向量维度投影、正负例定义、检索指标 |
| 文本生成 | 后端与产物流程、适用的 KL/CE | token 序列、词表或文本对齐、长度与序列 mask |

第一版只实现第一个真实需求；其余保留扩展边界，不先写没有调用者的投影网络、生成训练器或通用任务 DSL。不同 tokenizer 的逐 token KL 不天然成立，不能把候选 KL 直接搬去生成任务。

## 建议目录

以下为规划，不创建空目录或占位实现：

```text
lib/easy_ai/
|-- distillation/
|   |-- teacher.rb                 # 教师能力协议
|   |-- teachers/local_http.rb     # 本地推理服务，Faraday
|   |-- collector.rb               # 离线分批采集、重试、恢复
|   |-- artifact.rb                # 记录格式、校验、manifest
|   `-- losses/soft_targets.rb     # 数值稳定的温度 KL
`-- decision/
    |-- distillation_adapter.rb    # 候选请求、映射与监督检查
    `-- trainer.rb                 # 集成监督，复用现有训练与恢复

config/distillation/               # 教师/采集配置；学生模型仍归所属任务
benchmarks/distillation/            # 教师验收、教师/无教师对照
learning/distillation/             # 独立的小型教学演示
test/easy_ai/distillation/          # 公共模块测试
data/distillation/                 # 离线教师信号，已有 /data/ 忽略规则覆盖
runs/distillation/                 # 采集日志与报告，已有 /runs/ 规则覆盖
```

复用现有 `NN`、`Optim`、`Runtime`。初版不为蒸馏再复制一套 AdamW、GPU 回退、checkpoint 或 Decision Trainer。公共采集状态与学生优化器 checkpoint 是不同产物：前者跟踪哪些请求完成，后者跟踪模型训练更新；分别保持可恢复和可核验。

## 教师信号协议

每条产物保留任务标识、样本/源分组 ID、输入指纹、监督类型与数据；manifest 保存教师模型修订、量化、后端版本、提示模板、生成/评分参数、候选评分协议和数据 split 指纹。Decision 还需有规范化候选 ID 与**文本**指纹，按 ID/文本对齐而非数组下标；同一 ID 文本变化必须使旧缓存失效。

教师能力显式区分：最终标签、文本、完整候选 logits、已归一化候选分布、表示向量。后端不支持候选打分时，只能运行相应伪标签模式，不能把自由生成的“90%”当作真实概率，也不能把 top-k 中缺失选项当成零概率。

logits 与概率的格式、是否已施加温度必须明确；若只存概率，应记录生成温度和精度，不能重复温度处理。有效候选需完整覆盖，验证有限数、非负值、归一化和非空 mask。不同样本可有不同候选数量，padding 不参与 softmax、KL 或 reduction。

缓存键包括输入、任务适配器版本、教师及评分配置，不能只用原始文本。离线采集先写临时文件，完整验证后提交 manifest；恢复时校验已完成记录，错误输出明确记为失败，不静默变成标签。只对 train 采集用于学生训练的信号；教师验收的 development 数据与隔离 test 保持独立，同源改写和翻译不能跨 split。

## 训练与验收边界

首个软标签目标可采用：

```text
L = CE(gold, student_logits)
    + lambda * T^2 * KL(teacher_distribution_at_T || student_distribution_at_T)
```

KL 使用数值稳定的 log-softmax，教师信号不带梯度。每条样本在有效候选内求和、再按有效样本数平均，梯度累积也保持相同权重；T 和 lambda 随配置写入 checkpoint。T 是蒸馏温度，与推理后 calibration 温度分开。损失形式参考[知识蒸馏原论文](https://arxiv.org/abs/1503.02531)，具体混合权重仍需本项目验证。

只有硬伪标签时采用明确的 CE 模式；没有金标时不能虚构 `gold`，需单独标识未标注数据和权重。原有可信金标保留，教师冲突可审计。新增教师信号也是训练输入，指纹变化应阻止直接 resume。相同批次的完整训练和中断续训应可对照，候选重排后损失应一致。

验收分三层：公共模块测试数值/对齐/产物恢复；教师在固定开发任务上是否可靠；学生在隔离评估上是否受益。至少对比无教师硬标签与相同输入的教师监督，报告多种子、分语言、绑定/否定分项、学生成本和额外教师成本。JSON 结构化、可恢复和 loss 下降都不是蒸馏有效的充分条件。

6 GiB 本机优先离线运行教师，完成后退出，再训练学生；通过本地 HTTP 边界可保持 Ruby 主流程并使用已有原生推理后端。具体后端的评分能力、模型质量与实际显存要先实测。第一阶段不引入在线双模型共同训练、教师微调或 RL。

推荐顺序：冻结数据/验收协议 → 教师小规模验收 → 实现公共采集与产物 → Decision 适配与软标签 loss → 相同输入的学生对照。整体路线选择与既有失败证据见[迭代回顾](retrospective.md)。
