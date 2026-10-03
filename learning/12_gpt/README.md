# 12 · GPT：自回归文本训练与生成

状态：已有 `train.rb` 与 `lib/easy_ai_learning/gpt/` 的模型、配置、训练器和数据/batch helpers。先修：07、11。从仓库根目录执行：

```bash
bundle exec ruby learning/12_gpt/train.rb \
  --data data/learning/song.txt --tokenizer byte --iters 200 --device cpu
```

本地语料被忽略，必须先准备；使用 `--data` 或 `EASY_AI_DATA` 指定。历史说明见 `../LEGACY_README.md`。

学习顺序：tokenize → 窗口/右移 target → causal decoder → logits → next-token cross-entropy → AdamW 更新 → 逐步采样。现有组件有 LayerNorm、残差、attention/FFN dropout 和梯度裁剪，优化器对所有参数统一施加 weight decay。

待补课程实验：独立验证集与 perplexity、参数组、warmup/scheduler、上下文长度对照、temperature/top-k 等采样策略的受控比较、逐层诊断和可恢复 checkpoint。这些不是现有训练器已经完成的能力。训练 loss 下降或生成像语料不等于独立泛化或事实正确性。

结合 02 对照 SGD/AdamW；结合 03 检查 Embedding、attention 投影、FFN、归一化参数的不同尺度。结合 10 验证未来不可见、target 右移以及 padding/loss mask 的语义；目前 causal 组件未提供 padding mask 接口。

生成与重构对照见 [13](../13_generative/README.md)，迁移与微调见 [14](../14_transfer_learning/README.md)。[RL](../15_rl/README.md) 是奖励驱动的学习分支，先用小 MLP/环境理解，并不以 GPT 为先修。
