# AI 学习课程

这套课程按「数学与数据 → 最小神经网络 → 训练与诊断 → 表示学习与视觉 → 序列与语言 → 生成、迁移与决策 → 综合实验」组织。每章同时回答：模型怎么算、目标怎么定义、参数怎么学、结果怎么验证。编号是建议阅读顺序，分支之间可按先修关系选择。

课程目录已经建立；**课程规划完整不代表每章代码已经实现**。目前基础 NN 有完整小实验，GPT 有训练示例，Attention/Transformer 有共享组件，Tokenizer 有实现与历史实验；其余章节是明确待实现实验的大纲。

## 课程目录与先修关系

| 目录 | 核心内容 | 最小实验 / 学习产出 | 先修 | 当前状态 |
| --- | --- | --- | --- | --- |
| [00_foundations](00_foundations/README.md) | 张量、导数、概率、数据划分、经典 ML 基线 | 线性/逻辑回归、PCA 与指标计算 | 无 | 大纲 |
| [01_basic_nn](01_basic_nn/README.md) | 感知机、MLP、ReLU、链式法则 | 9 参数 XOR，手工梯度与 autograd 对照 | 00 | 已有可运行实验与图 |
| [02_training](02_training/README.md) | SGD → Momentum → Adam → AdamW；正则化与训练循环 | 同一 MLP 对照优化器、学习率、dropout | 01 | 大纲；SGD/AdamW 分别已有使用点 |
| [03_diagnostics](03_diagnostics/README.md) | 权重、激活、梯度、更新与泛化诊断 | 逐层统计，制造并修复训练故障 | 02 | 大纲；XOR 已能导出参数/loss |
| [04_autoencoder](04_autoencoder/README.md) | Auto-encoding、瓶颈、重构、稀疏与去噪 | 低维压缩和带遮挡输入的重构 | 01–03 | 大纲 |
| [05_cnn](05_cnn/README.md) | 卷积、共享参数、感受野、池化、LeNet | 小图像分类，对照 MLP/CNN | 01–03 | 大纲 |
| [06_resnet](06_resnet/README.md) | 深层优化、残差连接、BatchNorm | 同深度普通 CNN 与残差 CNN 对照 | 05 | 大纲 |
| [07_tokenizers](07_tokenizers/README.md) | 字符/字节/BPE、Embedding、数据泄漏 | 编解码往返、词表和序列长度比较 | 00–02 | 已有共享实现与历史实验 |
| [08_rnn](08_rnn/README.md) | RNN → LSTM/GRU、BPTT、梯度裁剪 | 序列规律预测与长距离记忆 | 01–03；文本任务加 07 | 大纲 |
| [09_seq2seq](09_seq2seq/README.md) | Encoder/Decoder、teacher forcing | 序列反转，训练与自由生成对照 | 08 | 大纲 |
| [10_attention](10_attention/README.md) | 加性注意力 → Q/K/V、多头、mask | 检索/对齐，注意力矩阵与未来泄漏验证 | 09 | 已有 causal attention 组件 |
| [11_transformer](11_transformer/README.md) | 位置编码、残差、LayerNorm、FFN | Encoder/Decoder 及位置/归一化消融 | 10；残差原理见 06 | 已有 block/FFN/位置组件 |
| [12_gpt](12_gpt/README.md) | Decoder-only、next-token、采样 | 小语料自回归训练与生成 | 07、11 | 已有训练示例 |
| [13_generative](13_generative/README.md) | VAE → GAN → Diffusion；BERT/MAE 遮挡重构选修 | 重构与生成对照、生成失败诊断 | 04；CNN 任务加 05；文本加 11–12 | 大纲 |
| [14_transfer_learning](14_transfer_learning/README.md) | 预训练、冻结、全量微调、LoRA、蒸馏 | 从头训练与迁移对照 | 02–03，加所选模型 | 大纲 |
| [15_rl](15_rl/README.md) | Bandit → MDP → Q-learning/DQN → 策略梯度/PPO | 奖励、回报、探索和策略评估 | 01–03；无需先学 GPT | 已有详细大纲 |
| [16_capstone](16_capstone/README.md) | 数据、模型、训练、诊断、推理完整闭环 | 可复现实验报告与模型复用 | 所选分支 | 大纲 |

建议先完成 00–03，再学简单的全连接 Autoencoder。视觉分支走 05–06；语言分支走 07–12。13 内的主题按任务选学；14 与 15 是迁移和交互学习分支，不表示它们是比 GPT 更大的网络。

## 模型与训练技术怎样一起学

| 技术 | 首次讲解 | 适合复习的模型与理由 |
| --- | --- | --- |
| SGD、Momentum、Adam、AdamW | 02，同一数据/MLP 对照 | 05–06 比较视觉任务；12 理解现有 AdamW 文本训练 |
| 初始化、学习率、scheduler、warmup | 02 原理、03 故障诊断 | 06 深层梯度传播；11–12 训练稳定性 |
| Dropout、weight decay、early stopping | 02，带独立验证集的 MLP | 04 防止仅记忆；11–12 注意力/FFN dropout |
| 输入 corruption / 随机遮挡 | 04 去噪 Autoencoder | 13 BERT/MAE 遮挡预测；是训练目标的一部分 |
| Padding mask / loss mask | 08–09 可变长批次 | 10–12：可见性和计分位置必须分别处理 |
| Causal mask | 10 | 12 禁止读取未来 token，训练和生成一致 |
| 数据标准化、BatchNorm、LayerNorm | 02 概念，03 观测 | 05–06 BatchNorm；11 LayerNorm，不能混称权重归一化 |
| 梯度裁剪、累积、混合精度 | 02 概念，08 裁剪实验 | 12 大一点的训练；先建立全精度正确性基线 |
| 参数组、冻结与低秩更新 | 14 | ResNet/GPT 迁移；结合 03 观察真正发生的更新 |

Mask 必须说明「遮什么、在哪一步、为什么」：dropout 的随机激活掩码、输入遮挡、attention 可见性掩码和 loss 掩码有不同语义，不能当作同一种正则化。

## 现有代码和入口

所有命令从仓库根目录执行。教学实现保持独立的 `EasyAILearning` 命名空间，生产库继续在 `lib/easy_ai/`。

```text
learning/
├── 00_foundations/ … 16_capstone/  章节 README；已有脚本随章节放置
├── 01_basic_nn/{logic,train,predict,plot}.rb
├── 07_tokenizers/{scratch,sentence_piece}.rb  历史实验
├── 12_gpt/train.rb
├── lib/easy_ai_learning/           共享实现按领域命名，不带课程编号
│   ├── basic_nn/                  逻辑、Torch MLP/SGD、手工梯度、报告/绘图
│   ├── attention/                 causal_self_attention.rb
│   ├── transformer/               block、feed_forward、positional_embeddings
│   ├── gpt/                       model、config、trainer、batch/text helpers
│   ├── tokenizers/                word/byte BPE、可选 Qwen 等
│   └── utils/                     tensor helpers
├── test/                          basic_nn、gpt、tokenizers
└── LEGACY_README.md                历史记录，保留原命令不作为当前入口
```

```bash
bundle exec ruby learning/01_basic_nn/logic.rb
bundle exec ruby learning/01_basic_nn/train.rb
bundle exec ruby learning/01_basic_nn/predict.rb
bundle exec ruby learning/01_basic_nn/plot.rb  # 需要 gnuplot
bundle exec ruby learning/12_gpt/train.rb \
  --data data/learning/song.txt --tokenizer byte --iters 200 --device cpu
bundle exec rake test:learning
```

XOR 默认优先 CUDA、不可用时回退 CPU；`--device cpu` 可强制 CPU。不需要下载模型或准备语料。默认 seed 1337 的已有 CUDA 记录为 436 步、最大绝对误差 0.009843、4/4 真值表预测正确；这是完整训练表上的拟合，不能称为泛化。细节、参数复用和现有图片见 [01](01_basic_nn/README.md)。

XOR 输出在忽略的 `runs/learning/basic_nn/logic-gates/`，包含推理参数、loss 和图；这些不是包含优化器状态的训练恢复 checkpoint。重跑会覆盖默认输出，比较实验应使用 `--output`。GPT 的本地语料在忽略的 `data/learning/`，通过 `--data` 或 `EASY_AI_DATA` 指定；新环境需自行准备。`bin/train_basic.rb` 已指向 `12_gpt/train.rb`。

本次目录迁移：`02_rnn → 08_rnn`、`03_seq2seq → 09_seq2seq`、`04_attention → 10_attention`、`05_transformer → 11_transformer`、`06_gpt → 12_gpt`、`07_rl → 15_rl`、`tokenizers → 07_tokenizers`。共享库命名、测试位置和现有产物路径不随章节编号移动。

## 每章的完成标准

每章从可手算的小例子开始，再提供 Ruby/Torch.rb 实验：先修与目标、forward 及张量形状、loss/backward、参数量、训练/推理差异、最小数据、可执行命令、学习曲线和失败案例。规划中的实验在脚本落地前不能标注为可运行。

训练实验需记录数据划分、种子、配置、更新次数和评估指标；比较时控制变量、使用多种子并报告波动。先验证一小批数据可拟合，再验证独立样本，最后做消融。有限真值表、训练集重构和训练 loss 都不能代替泛化评估。

图表优先 `unicode_plot`；需要发布的图再用 gnuplot。新增共享实现分别归入领域模块，例如 `autoencoder/`、`cnn/`、`resnet/`、`training/`、`diagnostics/`，在真正实现时建立并配必要测试。训练产物统一放忽略的 `runs/learning/<topic>/<experiment>/`。
