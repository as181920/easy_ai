# 07 · Tokenizer 与 Embedding

状态：已有 `lib/easy_ai_learning/tokenizers/` 的共享实现及测试；本章 `scratch.rb` 是历史 BPE 演示，`sentence_piece.rb` 是纯 Ruby 的 Unigram 近似，不是完整 SentencePiece 复现。独立统一课程实验待实现。先修：00–02。用于后续文本 RNN 和 GPT。

顺序：字符 → UTF-8 字节 → 高频相邻对/BPE → 编码/解码与词表存储 → token ID/Embedding → padding/特殊 token；Unigram 和外部 Qwen tokenizer 选修。

Tokenizer 的“训练”是学习词表/合并规则，Embedding 的训练是通过任务 loss 更新参数，两者不能混同。词表应只由训练语料确定；固定规则后再编码验证/测试。记录 Unicode、空白和特殊 token 策略，检查未知字符与边界情况。

计划比较同一中英文样本的 token 数、词表大小、可逆性、序列截断与 Embedding 参数量；不要只按 token 少来判断分词优劣。文本先做数据划分再训练 tokenizer，避免语料泄漏；字符序列实验也可先于复杂 BPE。

历史演示入口：`bundle exec ruby learning/07_tokenizers/scratch.rb`。共享 byte/word BPE 已用于 [GPT](../12_gpt/README.md)。
