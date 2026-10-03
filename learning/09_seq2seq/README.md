# 09 · Seq2seq：编码、解码与 teacher forcing

状态：课程大纲，尚无独立实现。先修：08。下一章：[Attention](../10_attention/README.md)。

先用 RNN encoder 把输入压进状态，再用 decoder 逐步输出。从短序列反转/复制任务开始，不先引入 attention。区分输入长度、输出长度、起止 token、右移目标与每步输出 logits。

Teacher forcing 在训练时使用真实前一步 token；生成时使用模型自己的输出。两种条件下误差会不同，必须记录自由生成的整序列正确率和 EOS 行为，不能只看 token loss。使用 padding/loss mask 后按有效 token 数归约，检查目标是否错移或意外包含答案。

计划实验比较长度与瓶颈容量，再在 10 加 cross-attention 对照。画 encoder/decoder 状态与预测，保留训练条件和推理条件的区别。Autoencoder 的重构任务与 Seq2seq 的序列转换共享编码/解码概念，但目标不必相同。
