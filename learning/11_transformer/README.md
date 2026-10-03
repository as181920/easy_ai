# 11 · Transformer：残差、位置、归一化

已实现可运行的最小教学实验。先修：10_attention；残差复习 06。 下一章：[12_gpt](../12_gpt/README.md)。

复用 10 的 MultiHead、06 的残差概念、02 的训练器，将 embedding/position、attention、FFN、LayerNorm 组成网络。

EncoderBlock 支持双向 pre-/post-LN；causal Block 是 pre-LN；EncoderDecoder 组合双向 encoder、因果 decoder 和 cross-attention。FFN 是每位置共享的 `D→2D/4D→D` 与 GELU。

```ruby
x = x + attention.call(layer_norm1.call(x))
x = x + feed_forward.call(layer_norm2.call(x))
```

LayerNorm 对每个样本/位置的特征维统计；BatchNorm 通常跨 batch/空间统计并维护 running buffers。当前 LayerNorm 的 train/eval 不切换 running statistics，dropout 则切换。不要把归一化误解为要求权重有固定分布。

四符号反转实验：Encoder 直接预测反序位置，比较 learned position、无 position、无残差、无归一化与 post-LN；完整 EncoderDecoder 另按 teacher forcing 训练并自由生成。无 position 的双向 encoder 是排列等变的，无法凭空知道索引顺序；测试以置换关系验证这一点。

Sinusoidal 提供位置公式数据，learned positions 用于默认网络；RoPE/ViT/相对位置作为后续阅读，不把本轮实验冒称完整实现这些变体。小任务/浅层中无残差或无归一化也可能很好，消融结果不能推出大模型不需要它们。

来源：[Transformer](https://arxiv.org/abs/1706.03762)、[LayerNorm](https://arxiv.org/abs/1607.06450)。

## 数据、训练与独立推理

从仓库根目录运行；安装依赖用 `bundle install`。涉及训练与模型推理时默认 `auto`：优先 CUDA，不可用时回退 CPU；也可显式 `--device cpu`。使用小规模合成数据，无模型下载。限制线程能避免 tiny tensor 的 CPU 线程开销：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/11_transformer/data.rb
bundle exec ruby learning/11_transformer/train.rb --steps 60
bundle exec ruby learning/11_transformer/predict.rb
bundle exec rake test:learning
```

`--seed` 改随机种子，`--output` 分开实验目录，训练还可用 `--device cpu/cuda/auto`。默认输出 `runs/learning/11_transformer/default/`，重跑会覆盖同名产物。`data.rb` 导出数据配方的样本用于查看；训练入口自行调用生成器，不依赖该 JSON 文件，训练中的特殊移位/遮挡在实验源码与实际 `data.json` 中记录。

推理 `--model PATH` 指向保存的模型。 `--input PATH` 可指定 `{"input": ...}` JSON；默认提供一个符合本章形状的示例。Seq2seq/EncoderDecoder 输出自由生成，GPT 输出 top-1 生成，其他模型输出 score/reconstruction；RL actor-critic 输出 policy logits 与 value。

## 结果与正确性

[实际运行记录](results.json) 保存 seed、步数、环境与指标；下图取自同一运行，不是验收阈值。

![本章实验结果](images/encoder-decoder-loss.svg)

[实验代码](../lib/easy_ai_learning/transformer/experiment.rb) 串起各步骤；共享数据在 [course/data.rb](../lib/easy_ai_learning/course/data.rb)。核心验证见 [测试](../test/course/attention_test.rb)，梯度对照另见 [derivatives_test.rb](../test/course/derivatives_test.rb)。测试只检查确定性公式、shape、mask、梯度、状态与参数更新；不训练到某个准确率或权重分布。

产物包括实际数据、JSON 推理状态、history、诊断与 SVG；本地完整参数/历史留在忽略的 runs 下，仓库只收录小型结果摘要与图。00/07 的非神经实验和 15 的交互训练输出格式按任务分别记录，不强行统一为分类 loss。
