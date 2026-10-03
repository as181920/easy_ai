# 11 · Transformer：把组件组成网络

状态：已有 `lib/easy_ai_learning/transformer/` 的 block、FFN、位置 Embedding，由 GPT 使用；完整独立 Encoder/Decoder 教学实验待实现。先修：10；残差可结合 06 理解。

顺序：token embedding + position → multi-head attention → residual → normalization → FFN → 多层堆叠；比较 Encoder 双向注意力、Decoder 因果注意力、Encoder–Decoder cross-attention。现有 `Block` 是 causal pre-LayerNorm block，不代表已经实现论文完整翻译模型。

从 06 复用 `x + F(x)`，解释 LayerNorm 与 CNN 中 BatchNorm 的统计维度和模式区别。比较 pre-/post-LN、绝对位置/正弦位置；RoPE、相对位置、ViT 作为完成基本模型后的选修，不要一次堆全部变体。

计划实验在小序列任务上逐项去掉位置、残差或归一化，观察顺序辨别、梯度传播、训练/验证表现；对照 09 的递归模型。训练先用 02 的简单正确循环，再比较 AdamW、warmup/schedule、dropout 和梯度裁剪。

原理来源：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)。下一章：[GPT](../12_gpt/README.md)，自监督双向遮挡预测见 [13](../13_generative/README.md)。
