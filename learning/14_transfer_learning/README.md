# 14 · 迁移、冻结、LoRA 与蒸馏

已实现可运行的最小教学实验。先修：02–03 与所选模型。 下一章：[15_rl](../15_rl/README.md)。

复用 02 的 MLP、优化器和保存格式，不下载外部模型。先在 96 个象限分类样本预训练，再用 32 个坐标平移样本做目标任务，独立验证 96 个。标签由原始坐标决定，平移的是观测输入，形成简单 domain shift。

对照：从头训练 → 冻结 hidden、只训 head → 冻结 hidden weight、解冻 bias/head → 全量微调 → 冻结 base 的低秩 head 更新。另用 width 4 的小 student 从预训练 teacher 的软目标蒸馏。Teacher 在源坐标上训练，所以它对目标域可能错误，蒸馏不是保证提升。

```text
LoRA: y=Wx+b+(α/r) B(Ax)
A 随机初始化，B=0；初始与 base 精确相同
merge: W_merged=W+(α/r)BA
```

LowRankLinear 与前面的 Linear 一样通过 call 使用；本章只将低秩适配加到分类 head，不声称实现了所有 Transformer 投影的 LoRA 框架。保存 base+A+B 可独立加载；测试比较 merged weight 与原 forward。

Distillation 使用 detached `softmax(teacher/T)`，student log-softmax，乘 T²，再与真实标签 CE 混合。这里计算 soft cross-entropy；它与 KL 差 teacher entropy 常数，gradient 等价，但数值不等同。

每个结果记录 trainable 参数量、每层实际更新与验证准确率；冻结层必须精确不变。CNN 迁移时还有 BatchNorm running statistics，冻结梯度并不等于冻结 buffer；本章用 MLP 避免隐藏该区别，后续扩展需显式处理。

## 数据、训练与独立推理

从仓库根目录运行；安装依赖用 `bundle install`。涉及训练与模型推理时默认 `auto`：优先 CUDA，不可用时回退 CPU；也可显式 `--device cpu`。使用小规模合成数据，无模型下载。限制线程能避免 tiny tensor 的 CPU 线程开销：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/14_transfer_learning/data.rb
bundle exec ruby learning/14_transfer_learning/train.rb --steps 60
bundle exec ruby learning/14_transfer_learning/predict.rb
bundle exec rake test:learning
```

`--seed` 改随机种子，`--output` 分开实验目录，训练还可用 `--device cpu/cuda/auto`。默认输出 `runs/learning/14_transfer_learning/default/`，重跑会覆盖同名产物。`data.rb` 导出数据配方的样本用于查看；训练入口自行调用生成器，不依赖该 JSON 文件，训练中的特殊移位/遮挡在实验源码与实际 `data.json` 中记录。

推理 `--model PATH` 指向保存的模型。 `--input PATH` 可指定 `{"input": ...}` JSON；默认提供一个符合本章形状的示例。Seq2seq/EncoderDecoder 输出自由生成，GPT 输出 top-1 生成，其他模型输出 score/reconstruction；RL actor-critic 输出 policy logits 与 value。

## 结果与正确性

[实际运行记录](results.json) 保存 seed、步数、环境与指标；下图取自同一运行，不是验收阈值。

![本章实验结果](images/full-loss.svg)

[实验代码](../lib/easy_ai_learning/transfer/experiment.rb) 串起各步骤；共享数据在 [course/data.rb](../lib/easy_ai_learning/course/data.rb)。核心验证见 [测试](../test/course/transfer_test.rb)，梯度对照另见 [derivatives_test.rb](../test/course/derivatives_test.rb)。测试只检查确定性公式、shape、mask、梯度、状态与参数更新；不训练到某个准确率或权重分布。

产物包括实际数据、JSON 推理状态、history、诊断与 SVG；本地完整参数/历史留在忽略的 runs 下，仓库只收录小型结果摘要与图。00/07 的非神经实验和 15 的交互训练输出格式按任务分别记录，不强行统一为分类 loss。
