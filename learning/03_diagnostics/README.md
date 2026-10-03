# 03 · 权重、激活、梯度与训练诊断

已实现可运行的最小教学实验。先修：02_training。 下一章：[04_autoencoder](../04_autoencoder/README.md)。

**没有通用的合理权重直方图；集中在零附近本身不是失败。** 初始化、稀疏任务、weight decay、归一化结构都可能产生集中分布。目标是稳定计算、有效学习与独立验证表现，不是把权重强行“拉散”。

本章复用 02 的同一数据与 MLP，分别运行基线、过强衰减、过小学习率。比较初始/最终逐层权重、梯度、实际更新、固定验证 batch 激活，以及隐藏矩阵的奇异值和有效秩。

| 观测 | 方法 | 解释限制 |
| --- | --- | --- |
| 权重 | 均值/std/RMS、分位数、极值、近零比例、逐张量直方图 | bias、norm scale、Embedding 不混为一个分布 |
| 激活 | 同样统计 + ReLU 零比例 | 很多零不等于所有单元在所有样本上死亡 |
| 梯度 | 非有限值、每层范数、missing 与 zero 区分 | 冻结或未参与目标也可能无梯度 |
| 实际更新 | `||θ_after-θ_before||/(||θ_before||+ε)` | 近零参数还须看绝对更新；AdamW 不等于 ηg |
| 矩阵结构 | 奇异值，`effective_rank=exp(-Σp log p)`，p 为归一化奇异值 | 低秩可能符合任务，无统一验收阈值 |
| 验证/输出 | 分类指标、失败样本、常量预测、输出置信度 | 训练 loss/权重图不能替代泛化 |

不同 ReLU 层配套乘/除正系数可保持函数而改变权重尺度。Xavier 的初始方差约 `2/(fan_in+fan_out)`，He 对 ReLU 约 `2/fan_in`，不是训练后必须达到的方差。依据：[Xavier](https://proceedings.mlr.press/v9/glorot10a.html)、[He](https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)。

```ruby
before = EasyAILearning::Diagnostics::Stats.snapshot(model)
# 一次或多次已正确计算的参数更新
weights = EasyAILearning::Diagnostics::Stats.model(model)
updates = EasyAILearning::Diagnostics::Stats.updates(model, before)
```

修复路径：不更新先查注册/冻结/计算图和标签；发散先查输入尺度、数值操作、步长，再用初始化、归一化、残差和必要的裁剪；训练好验证差再比较数据增强、decay/dropout、early stopping 和容量。一次改变一个变量，并复查验证集。当前实验直接展示过强 decay 与小步长；其他故障的确定性检测见测试。

## 数据、训练与独立推理

从仓库根目录运行；安装依赖用 `bundle install`。涉及训练与模型推理时默认 `auto`：优先 CUDA，不可用时回退 CPU；也可显式 `--device cpu`。使用小规模合成数据，无模型下载。限制线程能避免 tiny tensor 的 CPU 线程开销：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/03_diagnostics/data.rb
bundle exec ruby learning/03_diagnostics/train.rb --steps 60
bundle exec ruby learning/03_diagnostics/predict.rb
bundle exec rake test:learning
```

`--seed` 改随机种子，`--output` 分开实验目录，训练还可用 `--device cpu/cuda/auto`。默认输出 `runs/learning/03_diagnostics/default/`，重跑会覆盖同名产物。`data.rb` 导出数据配方的样本用于查看；训练入口自行调用生成器，不依赖该 JSON 文件，训练中的特殊移位/遮挡在实验源码与实际 `data.json` 中记录。

推理 `--model PATH` 指向保存的模型。 `--input PATH` 可指定 `{"input": ...}` JSON；默认提供一个符合本章形状的示例。Seq2seq/EncoderDecoder 输出自由生成，GPT 输出 top-1 生成，其他模型输出 score/reconstruction；RL actor-critic 输出 policy logits 与 value。

## 结果与正确性

[实际运行记录](results.json) 保存 seed、步数、环境与指标；下图取自同一运行，不是验收阈值。

![本章实验结果](images/baseline-weights.svg)

[实验代码](../lib/easy_ai_learning/diagnostics/experiment.rb) 串起各步骤；共享数据在 [course/data.rb](../lib/easy_ai_learning/course/data.rb)。核心验证见 [测试](../test/course/foundations_test.rb)，梯度对照另见 [derivatives_test.rb](../test/course/derivatives_test.rb)。测试只检查确定性公式、shape、mask、梯度、状态与参数更新；不训练到某个准确率或权重分布。

产物包括实际数据、JSON 推理状态、history、诊断与 SVG；本地完整参数/历史留在忽略的 runs 下，仓库只收录小型结果摘要与图。00/07 的非神经实验和 15 的交互训练输出格式按任务分别记录，不强行统一为分类 loss。
