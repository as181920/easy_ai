# 09 · Seq2seq 与 teacher forcing

已实现可运行的最小教学实验。先修：08_rnn。 下一章：[10_attention](../10_attention/README.md)。

复用 08 的 RNN cell：encoder 读完 source，decoder 从 encoder final state 开始。Embedding 共享；任务从 next-token 升到四符号序列反转。

输入 `[a,b,c,d]`；target `[d,c,b,a,EOS]`；训练 decoder input `[BOS,d,c,b,a]`。PAD=0、BOS=1、EOS=2、普通符号=3–6。当前数据固定长度，无需把 padding 当真 token。

```ruby
state = encode(source)
decoder_tokens.each_step do |previous_token|
  state = decoder.call(embedding.call(previous_token), state)
  logits = head.call(state)
end
```

Teacher forcing 只给真实前一个 token，不能把同位置 target 输入模型。推理使用上一步自己的 argmax；EOS 后输出 EOS，避免已结束样本继续生成随机词。默认输出最长五步；CLI 推理读保存模型，无训练动作。

报告 teacher-forced token accuracy、free-running token accuracy 和整序列正确率。前者可能掩盖连锁错误，所以必须自由生成。可变长任务需要 attention 可见性 mask 和 loss mask 分别处理；本章不声称固定长度实现已经支持任意 padding 训练。

核心 Model 后续增加 `attention: true`，但默认 false 不建立 Attention 组件，保持 09 的先修路径。10 用同一 encoder/decoder 对照，而不是重写另一个难以比较的模型。

## 数据、训练与独立推理

从仓库根目录运行；安装依赖用 `bundle install`。涉及训练与模型推理时默认 `auto`：优先 CUDA，不可用时回退 CPU；也可显式 `--device cpu`。使用小规模合成数据，无模型下载。限制线程能避免 tiny tensor 的 CPU 线程开销：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/09_seq2seq/data.rb
bundle exec ruby learning/09_seq2seq/train.rb --steps 60
bundle exec ruby learning/09_seq2seq/predict.rb
bundle exec rake test:learning
```

`--seed` 改随机种子，`--output` 分开实验目录，训练还可用 `--device cpu/cuda/auto`。默认输出 `runs/learning/09_seq2seq/default/`，重跑会覆盖同名产物。`data.rb` 导出数据配方的样本用于查看；训练入口自行调用生成器，不依赖该 JSON 文件，训练中的特殊移位/遮挡在实验源码与实际 `data.json` 中记录。

推理 `--model PATH` 指向保存的模型。 `--input PATH` 可指定 `{"input": ...}` JSON；默认提供一个符合本章形状的示例。Seq2seq/EncoderDecoder 输出自由生成，GPT 输出 top-1 生成，其他模型输出 score/reconstruction；RL actor-critic 输出 policy logits 与 value。

## 结果与正确性

[实际运行记录](results.json) 保存 seed、步数、环境与指标；下图取自同一运行，不是验收阈值。

![本章实验结果](images/seq2seq-loss.svg)

[实验代码](../lib/easy_ai_learning/seq2seq/experiment.rb) 串起各步骤；共享数据在 [course/data.rb](../lib/easy_ai_learning/course/data.rb)。核心验证见 [测试](../test/course/sequence_test.rb)，梯度对照另见 [derivatives_test.rb](../test/course/derivatives_test.rb)。测试只检查确定性公式、shape、mask、梯度、状态与参数更新；不训练到某个准确率或权重分布。

产物包括实际数据、JSON 推理状态、history、诊断与 SVG；本地完整参数/历史留在忽略的 runs 下，仓库只收录小型结果摘要与图。00/07 的非神经实验和 15 的交互训练输出格式按任务分别记录，不强行统一为分类 loss。
