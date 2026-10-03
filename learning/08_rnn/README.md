# 08 · RNN → LSTM/GRU 与 BPTT

已实现可运行的最小教学实验。先修：01–03；文本加 07。 下一章：[09_seq2seq](../09_seq2seq/README.md)。

把前面的 Linear 层放进时间循环；同一参数在各时间步共享。RNN、LSTM、GRU 的门控公式直接实现，而不是把内置 cell 当黑盒。

```text
RNN: h'=tanh(Wx+Uh+b)
LSTM: i,f,o=sigmoid(gates), g=tanh(candidate)
      c'=f*c+i*g; h'=o*tanh(c')
GRU: r,z=sigmoid(gates); n=tanh(W_n x+b_n+r*(U_n h+b_h))
     h'=(1-z)*n+z*h
```

GRU 使用 PyTorch 常见 reset-after 公式；一些教材的 reset-before 变体不同。每个 Linear 有自己的 bias，RNN/LSTM 的 input/recurrent bias 在求和后起作用。

最小实验是六符号循环 next-token；再做延迟复制：第一步给 0–5，后面全为 filler 6，最后一刻预测首符号。默认训练长度 10，同时观察长度 20；每种 cell 用独立但相同 seed 的初始化比较。短运行的好坏不用于测试断言。

BPTT 由 Torch 跟踪 Ruby 展开的循环；`truncate:` 在时间边界 detach，阻断跨边界梯度。`lengths:` 对越界位置保持前一 hidden/cell state，不能仅靠 padding loss mask 假装状态没有更新。每个独立 forward 默认初始化状态；连续流才显式传入状态。

复用 02 的梯度裁剪、优化器和状态保存。输出 token accuracy、不同长度记忆表现、训练/验证曲线、裁剪前后范数。测试用手算门、有限差分和首步 Embedding 梯度验证完整/截断 BPTT。

## 数据、训练与独立推理

从仓库根目录运行；安装依赖用 `bundle install`。涉及训练与模型推理时默认 `auto`：优先 CUDA，不可用时回退 CPU；也可显式 `--device cpu`。使用小规模合成数据，无模型下载。限制线程能避免 tiny tensor 的 CPU 线程开销：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/08_rnn/data.rb
bundle exec ruby learning/08_rnn/train.rb --steps 60
bundle exec ruby learning/08_rnn/predict.rb
bundle exec rake test:learning
```

`--seed` 改随机种子，`--output` 分开实验目录，训练还可用 `--device cpu/cuda/auto`。默认输出 `runs/learning/08_rnn/default/`，重跑会覆盖同名产物。`data.rb` 导出数据配方的样本用于查看；训练入口自行调用生成器，不依赖该 JSON 文件，训练中的特殊移位/遮挡在实验源码与实际 `data.json` 中记录。

推理 `--model PATH` 指向保存的模型。 `--input PATH` 可指定 `{"input": ...}` JSON；默认提供一个符合本章形状的示例。Seq2seq/EncoderDecoder 输出自由生成，GPT 输出 top-1 生成，其他模型输出 score/reconstruction；RL actor-critic 输出 policy logits 与 value。

## 结果与正确性

[实际运行记录](results.json) 保存 seed、步数、环境与指标；下图取自同一运行，不是验收阈值。

![本章实验结果](images/lstm-memory-loss.svg)

[实验代码](../lib/easy_ai_learning/rnn/experiment.rb) 串起各步骤；共享数据在 [course/data.rb](../lib/easy_ai_learning/course/data.rb)。核心验证见 [测试](../test/course/sequence_test.rb)，梯度对照另见 [derivatives_test.rb](../test/course/derivatives_test.rb)。测试只检查确定性公式、shape、mask、梯度、状态与参数更新；不训练到某个准确率或权重分布。

产物包括实际数据、JSON 推理状态、history、诊断与 SVG；本地完整参数/历史留在忽略的 runs 下，仓库只收录小型结果摘要与图。00/07 的非神经实验和 15 的交互训练输出格式按任务分别记录，不强行统一为分类 loss。
