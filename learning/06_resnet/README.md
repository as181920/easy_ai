# 06 · ResNet：残差与深层优化

已实现可运行的最小教学实验。先修：05_cnn。 下一章：[07_tokenizers](../07_tokenizers/README.md)。

复用 CNN 的卷积、BatchNorm、全局池化和同一图像数据，改变深层 block 的信息流：`y=ReLU(x+F(x))`，形状不同时先用 1×1 projection/stride 对齐。

本章普通深网络与残差网络各堆三块，每块两个 3×3 卷积；相同通道、深度、初始权重和 BatchNorm 设置。另做无归一化残差对照。这里的普通分支确实去掉相加，不能仅把 shortcut 参数设零就宣称等价。

```ruby
residual = norm2.call(conv2.call(relu(norm1.call(conv1.call(x)))))
y = relu(shortcut.call(x) + residual)
```

basic block、projection、`1×1→3×3→1×1` bottleneck 和 pre-activation 块均有实现。后三者有形状/直通梯度演示与测试；默认训练对照使用 basic block。Pre-activation 将 BN/ReLU 放在卷积前，最后相加后不额外 ReLU，因此零 residual 可保留负输入；原 basic block 的零 residual 仍经过最后 ReLU。

残差给梯度提供直接路径，但不保证任何训练都无消失/爆炸或验证更好。小图形任务很容易，全部满分不能证明残差没有价值，也不能证明它总是提升。逐层统计与同预算比较应一起读。残差思想在 11 的 Transformer 复用；BatchNorm 与 LayerNorm 的维度和模式不同。

来源：[ResNet](https://arxiv.org/abs/1512.03385)。

## 数据、训练与独立推理

从仓库根目录运行；安装依赖用 `bundle install`。涉及训练与模型推理时默认 `auto`：优先 CUDA，不可用时回退 CPU；也可显式 `--device cpu`。使用小规模合成数据，无模型下载。限制线程能避免 tiny tensor 的 CPU 线程开销：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/06_resnet/data.rb
bundle exec ruby learning/06_resnet/train.rb --steps 60
bundle exec ruby learning/06_resnet/predict.rb
bundle exec rake test:learning
```

`--seed` 改随机种子，`--output` 分开实验目录，训练还可用 `--device cpu/cuda/auto`。默认输出 `runs/learning/06_resnet/default/`，重跑会覆盖同名产物。`data.rb` 导出数据配方的样本用于查看；训练入口自行调用生成器，不依赖该 JSON 文件，训练中的特殊移位/遮挡在实验源码与实际 `data.json` 中记录。

推理 `--model PATH` 指向保存的模型。 `--input PATH` 可指定 `{"input": ...}` JSON；默认提供一个符合本章形状的示例。Seq2seq/EncoderDecoder 输出自由生成，GPT 输出 top-1 生成，其他模型输出 score/reconstruction；RL actor-critic 输出 policy logits 与 value。

## 结果与正确性

[实际运行记录](results.json) 保存 seed、步数、环境与指标；下图取自同一运行，不是验收阈值。

![本章实验结果](images/residual-loss.svg)

[实验代码](../lib/easy_ai_learning/resnet/experiment.rb) 串起各步骤；共享数据在 [course/data.rb](../lib/easy_ai_learning/course/data.rb)。核心验证见 [测试](../test/course/vision_test.rb)，梯度对照另见 [derivatives_test.rb](../test/course/derivatives_test.rb)。测试只检查确定性公式、shape、mask、梯度、状态与参数更新；不训练到某个准确率或权重分布。

产物包括实际数据、JSON 推理状态、history、诊断与 SVG；本地完整参数/历史留在忽略的 runs 下，仓库只收录小型结果摘要与图。00/07 的非神经实验和 15 的交互训练输出格式按任务分别记录，不强行统一为分类 loss。
