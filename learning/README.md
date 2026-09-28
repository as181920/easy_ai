# 学习代码

原有 GPT、word BPE、旧 byte BPE、Qwen tokenizer 实验保留在此，统一使用 `EasyAILearning`，不由正式库自动加载。正式维护的候选概率能力位于 `lib/easy_ai/decision/`；教学目录可实验性修改而不改变其 API。

| 原位置 | 当前位置 |
| --- | --- |
| `lib/easy_ai/{models,modules,data,trainers,utils}/` | `learning/lib/easy_ai_learning/` 对应目录 |
| `lib/easy_ai/config.rb` | `learning/lib/easy_ai_learning/config.rb` |
| 原 tokenizer 实验与测试 | `learning/lib/easy_ai_learning/tokenizers/`、`learning/test/` |
| `sentence_piece_tokenizer.rb` | `learning/tokenizers/sentence_piece.rb` |
| 本地 `temp.rb` 实验 | `learning/tokenizers/scratch.rb` |
| 原 GPT 训练入口 | `learning/transformer/train.rb`；`bin/train_basic.rb` 保留兼容转发 |
| 原 README | `LEGACY_README.md`，历史说明，旧命名不再直接适用 |
| 原 `data/*.txt` 学习数据 | `data/learning/`，继续由 Git 忽略 |

```bash
bundle exec rake test:learning
bundle exec ruby -Ilib -Ilearning/lib -reasy_ai_learning -e 'puts EasyAILearning::Models::GPT'
bundle exec ruby learning/transformer/train.rb --data data/learning/xiaojing.txt --tokenizer byte --device cpu
```

教学训练默认读取 `data/learning/`，可通过 `--data` 或 `EASY_AI_DATA` 指定其他文件或目录。该目录是本地学习数据，不随 Git 分发；新环境需自行准备语料。

旧 GPT 示例仍可选择预训练 Qwen tokenizer，这是保留的历史教学实验；新的 Decision 数据、分词和权重训练流程不依赖它。

建议按实际依赖阅读正式实现：

```text
tokenizers/byte_bpe
  -> decision/data/{example,collator,masking}
  -> nn/{rotary_position,attention,feed_forward,encoder_block}
  -> decision/{encoder,interaction_block,choice_model}
  -> optim/adamw -> decision/{trainer,checkpoint}
  -> decision/{calibrator,predictor,evaluator}
  -> decision/growth/{add_block,widen_ffn,controller}
```

对应测试验证 padding 隔离、可学习性、优化器数值、续训一致性、扩容保持原函数和回滚等性质。阅读时建议先运行 tiny 测试，再用 smoke 配置串起公开数据流程，最后运行 small 配置。无需为教学目的另抄一套相同生产算法；这里保留独立的历史实验，正式算法本身也应可读。

排查“loss 下降但否定仍错误”时，可从 [关系学习实验](../docs/decision/relations.md)开始：`relation_corpus` 生成可验证的布尔标签，`pair_sampler` 配对采样，`relation_evaluation` 测量反事实变化；实验编排入口在 `benchmarks/decision/relations.rb`。
