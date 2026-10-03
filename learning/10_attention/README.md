# 10 · Attention：对齐、检索与 mask

已实现可运行的最小教学实验。先修：09_seq2seq。 下一章：[11_transformer](../11_transformer/README.md)。

先用 additive attention 理解 query 检索 encoder states，再学 scaled dot-product、多头和 self/cross attention。复用 09 的编码/解码状态，`attention:true` 时将每步 decoder state 与检索 context 拼接并融合。

`A=softmax(QKᵀ/sqrt(d_k))`，`output=A V`；heads 将特征维分组再拼接，经 output projection。softmax 前看 score；dropout 前的 A 才具有每行和为 1 的概率解释。注意力图不自动构成因果解释。

| Mask | 作用位置 | 正确性 |
| --- | --- | --- |
| Padding | attention score 的无效 key | 改 padding 内容不改有效结果 |
| Causal | softmax 前遮未来 key | 改未来 token 不改过去 logits |
| Loss | target loss | 无效 target 不计分，按有效位置数归约 |
| Dropout | 激活或 softmax 后 attention 权重 | 训练随机，eval 关闭；不是可见性约束 |

```ruby
scores = matmul(q, k.transpose(-2, -1)) / Math.sqrt(head_dim)
scores = scores.masked_fill(valid.logical_not, -Float::INFINITY)
weights = softmax(scores, dim: -1)
output = matmul(weights, v)
```

MultiHead 支持 `[B,K]` padding key mask，true 表示可见；query 可以与 memory 长度不同。CausalSelfAttention 复用它并强制相等长度。若某 query 全无可见 key，显式报错，避免 softmax 全 -Inf。无效 query 的输出由调用者处理；不要仅提供 key mask 就说 query/loss 也忽略 padding。

实验比较 fixed-state 与 cross-attention 的反转任务，保存 alignment；另用改未来值的精确演示显示过去输出差为零。测试还用 identity projections 手算 softmax/加权和。来源：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

## 数据、训练与独立推理

从仓库根目录运行；安装依赖用 `bundle install`。涉及训练与模型推理时默认 `auto`：优先 CUDA，不可用时回退 CPU；也可显式 `--device cpu`。使用小规模合成数据，无模型下载。限制线程能避免 tiny tensor 的 CPU 线程开销：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/10_attention/data.rb
bundle exec ruby learning/10_attention/train.rb --steps 60
bundle exec ruby learning/10_attention/predict.rb
bundle exec rake test:learning
```

`--seed` 改随机种子，`--output` 分开实验目录，训练还可用 `--device cpu/cuda/auto`。默认输出 `runs/learning/10_attention/default/`，重跑会覆盖同名产物。`data.rb` 导出数据配方的样本用于查看；训练入口自行调用生成器，不依赖该 JSON 文件，训练中的特殊移位/遮挡在实验源码与实际 `data.json` 中记录。

推理 `--model PATH` 指向保存的模型。 `--input PATH` 可指定 `{"input": ...}` JSON；默认提供一个符合本章形状的示例。Seq2seq/EncoderDecoder 输出自由生成，GPT 输出 top-1 生成，其他模型输出 score/reconstruction；RL actor-critic 输出 policy logits 与 value。

## 结果与正确性

[实际运行记录](results.json) 保存 seed、步数、环境与指标；下图取自同一运行，不是验收阈值。

![本章实验结果](images/cross-attention-loss.svg)

[实验代码](../lib/easy_ai_learning/attention/experiment.rb) 串起各步骤；共享数据在 [course/data.rb](../lib/easy_ai_learning/course/data.rb)。核心验证见 [测试](../test/course/attention_test.rb)，梯度对照另见 [derivatives_test.rb](../test/course/derivatives_test.rb)。测试只检查确定性公式、shape、mask、梯度、状态与参数更新；不训练到某个准确率或权重分布。

产物包括实际数据、JSON 推理状态、history、诊断与 SVG；本地完整参数/历史留在忽略的 runs 下，仓库只收录小型结果摘要与图。00/07 的非神经实验和 15 的交互训练输出格式按任务分别记录，不强行统一为分类 loss。
