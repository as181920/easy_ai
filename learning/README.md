# AI 学习课程

这套课程按「数学与数据 → 最小神经网络 → 训练与诊断 → 表示学习与视觉 → 序列与语言 → 生成、迁移与决策 → 综合实验」组织。每章同时回答：模型怎么算、目标怎么定义、参数怎么学、结果怎么验证。编号是建议阅读顺序，分支之间可按先修关系选择。

00–16 现在都有可执行的数据/训练实验与学习文档。01 保留原来的完整 XOR 教学；后续章复用其基础思路，再复用训练、诊断、卷积、递归状态和注意力组件。默认训练与神经网络推理使用 `auto`，优先 CUDA、不可用时回退 CPU；`--device cpu` 可显式选择 CPU。经典 Ruby 数学计算和 Tokenizer 规则学习本身不依赖 GPU。

每章提供小规模合成任务、可复现数据导出、独立推理、真实结果 JSON 和图。大数据基准、完整官方 BERT/MAE 配置、RoPE/ViT 等只作阅读扩展，默认实验是可手算/易检查的最小版本。

打印阅读用 [全课程精简打印版](../docs/learning-course-print.md)：核心概念、公式与必要代码，无运行日志和大图。

## 课程目录与先修关系

| 目录 | 核心内容 | 最小实验 / 学习产出 | 先修 | 实现 |
| --- | --- | --- | --- | --- |
| [00_foundations](00_foundations/README.md) | 张量、导数、概率、数据划分、经典 ML 基线 | 线性/逻辑回归、PCA 与指标计算 | 无 | 已实现 |
| [01_basic_nn](01_basic_nn/README.md) | 感知机、MLP、ReLU、链式法则 | 9 参数 XOR，手工梯度与 autograd 对照 | 00 | 已有可运行实验与图 |
| [02_training](02_training/README.md) | SGD → Momentum → Adam → AdamW；正则化与训练循环 | 同一 MLP 对照优化器、学习率、dropout | 01 | 优化器、训练控制与对照实验 |
| [03_diagnostics](03_diagnostics/README.md) | 权重、激活、梯度、更新与泛化诊断 | 逐层统计，制造并修复训练故障 | 02 | 逐层统计、直方图、奇异值与故障对照 |
| [04_autoencoder](04_autoencoder/README.md) | Auto-encoding、瓶颈、重构、稀疏与去噪 | 低维压缩和带遮挡输入的重构 | 01–03 | 已实现 |
| [05_cnn](05_cnn/README.md) | 卷积、共享参数、感受野、池化、LeNet | 小图像分类，对照 MLP/CNN | 01–03 | 已实现 |
| [06_resnet](06_resnet/README.md) | 深层优化、残差连接、BatchNorm | 同深度普通 CNN 与残差 CNN 对照 | 05 | 已实现 |
| [07_tokenizers](07_tokenizers/README.md) | 字符/字节/BPE、Embedding、数据泄漏 | 编解码往返、词表和序列长度比较 | 00–02 | 可运行字符/字节/BPE 与 Embedding 实验 |
| [08_rnn](08_rnn/README.md) | RNN → LSTM/GRU、BPTT、梯度裁剪 | 序列规律预测与长距离记忆 | 01–03；文本任务加 07 | 已实现 |
| [09_seq2seq](09_seq2seq/README.md) | Encoder/Decoder、teacher forcing | 序列反转，训练与自由生成对照 | 08 | 已实现 |
| [10_attention](10_attention/README.md) | 加性注意力 → Q/K/V、多头、mask | 检索/对齐，注意力矩阵与未来泄漏验证 | 09 | Additive、多头、mask 与对照实验 |
| [11_transformer](11_transformer/README.md) | 位置编码、残差、LayerNorm、FFN | Encoder/Decoder 及位置/归一化消融 | 10；残差原理见 06 | Encoder、Decoder、消融和自由生成 |
| [12_gpt](12_gpt/README.md) | Decoder-only、next-token、采样 | 小语料自回归训练与生成 | 07、11 | 小语言实验；保留自定义语料入口 |
| [13_generative](13_generative/README.md) | VAE → GAN → Diffusion；BERT/MAE 遮挡重构选修 | 重构与生成对照、生成失败诊断 | 04；CNN 任务加 05；文本加 11–12 | 已实现 |
| [14_transfer_learning](14_transfer_learning/README.md) | 预训练、冻结、全量微调、LoRA、蒸馏 | 从头训练与迁移对照 | 02–03，加所选模型 | 已实现 |
| [15_rl](15_rl/README.md) | Bandit → MDP → Q-learning/DQN → 策略梯度/PPO | 奖励、回报、探索和策略评估 | 01–03；无需先学 GPT | Bandit/Q/DQN/REINFORCE/AC/PPO |
| [16_capstone](16_capstone/README.md) | 数据、模型、训练、诊断、推理完整闭环 | 可复现实验报告与模型复用 | 所选分支 | 已实现 |

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
bundle exec ruby learning/12_gpt/train_text.rb \
  --data data/learning/song.txt --tokenizer byte --iters 200 --device auto
bundle exec rake test:learning
```

XOR 默认优先 CUDA、不可用时回退 CPU；`--device cpu` 可强制 CPU。不需要下载模型或准备语料。默认 seed 1337 的已有 CUDA 记录为 436 步、最大绝对误差 0.009843、4/4 真值表预测正确；这是完整训练表上的拟合，不能称为泛化。细节、参数复用和现有图片见 [01](01_basic_nn/README.md)。

XOR 输出在忽略的 `runs/learning/basic_nn/logic-gates/`，包含推理参数、loss 和图；这些不是包含优化器状态的训练恢复 checkpoint。重跑会覆盖默认输出，比较实验应使用 `--output`。GPT 的本地语料在忽略的 `data/learning/`，通过 `--data` 或 `EASY_AI_DATA` 指定；新环境需自行准备。`bin/train_basic.rb` 已指向 `12_gpt/train.rb`。

本次目录迁移：`02_rnn → 08_rnn`、`03_seq2seq → 09_seq2seq`、`04_attention → 10_attention`、`05_transformer → 11_transformer`、`06_gpt → 12_gpt`、`07_rl → 15_rl`、`tokenizers → 07_tokenizers`。共享库命名、测试位置和现有产物路径不随章节编号移动。

## 每章的完成标准

每章从可手算的小例子开始，提供 Ruby/Torch.rb 实验：先修与目标、forward 及张量形状、loss/backward、参数量、训练/推理差异、最小数据、可执行命令、学习曲线和失败案例。最小可执行任务与仅供阅读的结构扩展分别注明。

训练实验需记录数据划分、种子、配置、更新次数和评估指标；比较时控制变量、使用多种子并报告波动。先验证一小批数据可拟合，再验证独立样本，最后做消融。有限真值表、训练集重构和训练 loss 都不能代替泛化评估。

XOR 终端图使用 `unicode_plot`，发布图使用 gnuplot；新增课程直接生成 SVG，无需额外绘图依赖。共享实现归入 `foundations/`、`training/`、`diagnostics/`、`autoencoder/`、`cnn/`、`resnet/`、`rnn/`、`seq2seq/`、`generative/`、`transfer/`、`rl/` 等领域模块；章节入口串起实验。训练产物统一放忽略的 `runs/learning/<topic>/<experiment>/`。

## 一条可执行的学习路径

```bash
# 只安装现有 Gemfile 依赖；不用新增下载语料或预训练模型
bundle install
bundle exec ruby learning/run_all.rb
# 显式 CPU 对照（单元测试也固定 CPU，不依赖 GPU 硬件）
bundle exec ruby learning/run_all.rb --device cpu --output runs/learning/cpu-course
bundle exec rake test:learning
```

`run_all.rb` 按 00–16 顺序导出数据和训练；默认每个对照模型 60 次 full-batch 更新，RL 为 60 episodes，XOR 保留原来的 10,000 次更新上限/误差目标。后续章节反复复用前章组件。默认输出每章 `runs/learning/<chapter>/default/`，合并摘要在 `runs/learning/run-summary.json`。完整重跑是教学实验，不是单元测试；训练结果不确定时仍保留实际记录。

每章可独立执行 `data.rb`、`train.rb`、`predict.rb`。新章节推理用 `--model PATH` 和 `--input JSON_FILE`，默认提供合法的小输入；00 用 `--value`，07 用 `--text`，01 保留原有参数。神经推理也支持 `--device`。设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` 可减少小 CPU 张量的线程开销；全课程 runner 会设置。

核心测试检查手算/闭式公式、有限差分/autograd、卷积、门控、BPTT detach、mask/未来泄漏、冻结/LoRA、GAN 梯度隔离、DDPM/GAE/PPO 目标、精确参数更新和状态往返；不以训练准确率、收敛步数、loss 必须下降或“权重长得合理”为通过条件。

通用教学 optimizer/Loop 自行支持 JSON 状态。Loop 的 per-update seed、相同 full-batch 数据/objective 与固定 total_steps 支持精确恢复；其约定不覆盖历史 Torch Trainer 或 RL 环境/rollout/replay 的任意时点恢复。EarlyStopping 保存最佳模型用于推理，其末尾 optimizer 状态不是最佳时刻的状态。

原始模型/优化器参数与完整数据只保存在忽略的 runs 下。各章 `results.json` 和 `images/` 是本轮真实运行的小型参考快照，含设备、seed 和步数；阅读结果时保留样本与任务的范围，不把简单合成数据上的满分当作真实 AI 能力证明。

## CUDA 环境记录

本轮 00–16 默认 `auto` 的实际实验全部完成；有神经张量的实验使用 CUDA，纯 Ruby 数学/Tokenizer 规则仍由 Ruby 计算。工具沙箱本身看不到驱动，因此 CPU 单元测试与沙箱外 CUDA 实验分别验证。库不可用/初始化失败时遵循现有 DevicePolicy 回退 CPU；本轮未将任意运行时错误吞掉并伪装成 CUDA 成功。

本机默认系统 cuDNN 在 CNN 路径报 `Invalid handle / cublasLtGetVersion`；选用本机已有的兼容 cuDNN 9 库后，CNN forward/backward 与全部 CUDA 实验通过。无需改变系统安装。需要同样配置时可用：

```bash
bundle exec ruby learning/run_all.rb --device auto \
  --cudnn-dir /home/andersen/Installed/libtorch-2.10.0-cu128/lib \
  --output runs/learning/cuda-course
```

`--cudnn-dir` 仅将该目录的 `libcudnn*.so.9` 链接到本次输出的 `runtime/cudnn/`，在子进程前置兼容库路径并保留原 LD_LIBRARY_PATH，避免同时替换其他 LibTorch 库。它是此机器的可选运行设置，不是所有设备的默认要求；兼容环境无需传入。独立推理也要使用相同兼容库环境。`--chapter 05_cnn` 可只跑一章。

[显式数值与状态实现](lib/easy_ai_learning/training/optimizer.rb) 中的 `kind`、RNN 的 `kind`、MLP 的 `activation`、经典集成的 `kind` 以及设备 `requested` 都兼容 string/symbol；未知枚举报 ArgumentError。[选项契约测试](test/course/options_test.rb) 验证两种写法相同。

覆盖查看：`bundle exec ruby learning/verify.rb --coverage`，报告在忽略的 `tmp/learning/coverage.json`。覆盖率仅用于发现漏测分支，核心逻辑的正确性依据仍是手算公式、有限差分和行为契约。
