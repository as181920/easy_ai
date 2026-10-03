# 10 · Attention 与不同的 mask

状态：已有 `lib/easy_ai_learning/attention/causal_self_attention.rb`，由 GPT 使用；独立推导、cross-attention 和可视化实验待实现。先修：09。

顺序：Seq2seq 加性 attention → query/key/value → scaled dot-product `softmax(QKᵀ/sqrt(d_k))V` → cross-/self-attention → multi-head → causal attention。手算一个微型矩阵，标出 batch/head/query/key 维度。注意力分布不是模型参数，也不自动构成因果解释。参考：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

| Mask | 遮什么 | 应做的验证 |
| --- | --- | --- |
| Padding attention mask | 无效 key 等位置的可见性 | 改变 padding 内容不改变有效输出；无效 query 输出按约定处理 |
| Causal mask | 当前 query 后面的未来 key | 改变未来输入不影响过去 logits |
| Loss mask | 不计分的 target 位置 | 改变无效标签不改变 loss，归约分母为有效位置数 |
| Dropout mask | attention 权重或激活的随机丢弃 | 训练有随机性，评估关闭；与可见性约束分别验证 |

可见性约束通常加在 softmax 前的 score 上；不能简单把 softmax 后的概率置零当作同一归一化。避免某个 query 全被遮挡而产生未定义/非有限结果，明确 mask 布尔语义和广播 shape。

现有组件仅有 causal mask 与 attention dropout，没有 padding mask 接口；不能将其描述为已支持任意变长/padding 批次。Dropout 后 attention 权重的单次行和不必为 1，统计与可视化要注明是在 dropout 前还是后。

计划实验：序列检索/对齐，绘制注意力矩阵，比较无 attention 与 attention 的 Seq2seq；单独验证未来泄漏和 padding 行为。下一章：[Transformer](../11_transformer/README.md)。
