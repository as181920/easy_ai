# 02 · 训练基础：从 SGD 到 AdamW

状态：课程大纲；[XOR](../01_basic_nn/README.md) 已使用 SGD，[GPT](../12_gpt/README.md) 已使用 AdamW，独立对照实验尚未实现。先修：01。下一章：[训练诊断](../03_diagnostics/README.md)。

## 先理解训练循环

数据/目标 → forward → loss → 清空旧梯度 → backward → 观测或裁剪梯度 → optimizer step → 验证。明确 batch、epoch、step、shuffle、梯度累积和 batch loss 的归约方式；分别设置训练模式与评估模式，并在推理时关闭梯度记录。

第一实验沿用 XOR 手算一步 SGD。第二实验使用有独立验证集的合成非线性分类数据和稍宽的 MLP；固定划分、结构、初始化、batch 顺序与更新预算，再做逐项对照。完整 XOR 表太小，不能拿 dropout 后的表现推断一般正则化效果。

## 优化器的递进

| 方法 | 核心机制 | 实验关注点 |
| --- | --- | --- |
| SGD | `θ ← θ - ηg` | 学习率太小/太大；随机批次与 full batch |
| Momentum | 累积梯度方向 | 振荡与收敛；明确使用哪种动量公式 |
| Adam | 梯度一阶/二阶矩、偏差修正 | 坐标自适应尺度、epsilon、早期更新 |
| AdamW | Adam 更新与权重衰减分离 | 衰减强度和学习率的联动；参数组 |

示意：`θ ← (1 - ηλ)θ - η * m_hat / (sqrt(v_hat) + ε)`。这里 λ 是 decoupled weight decay，不是把 `λθ` 加进 Adam 的梯度；Adam 的 L2 penalty 与 AdamW 一般不等价。依据：[AdamW 原论文](https://arxiv.org/abs/1711.05101)。

优化器没有固定冠军。先用同预算比较，再给每种方法公平的学习率搜索预算；记录训练与验证指标、多种子波动、耗时和参数更新量，不能只比较一个默认学习率的最后一次训练 loss。

## 控制训练与泛化

1. 初始化：打破对称性，配合层宽和激活选择 Xavier/He；零 bias 可以合理，所有隐藏权重相同则会妨碍学习。
2. 学习率：固定步长 → 衰减 → warmup + schedule；记录实际学习率。增加复杂技巧前保留简单基线。
3. Weight decay/L2、dropout、early stopping、数据增强分别做消融；依据独立验证集选配置，测试集留到最终评估。
4. 标准化输入；理解 BatchNorm/LayerNorm 处理的是激活统计，不是要求权重呈某种直方图。
5. 梯度裁剪限制梯度范数；不能替代查找梯度爆炸原因。累积时按真实样本/token 数正确缩放 loss；混合精度需理解 loss scaling 与数值范围。
6. 保存推理权重与可恢复训练 checkpoint 的区别：后者还需优化器、scheduler、步数、配置及必要的 RNG/数据迭代状态。

Dropout 训练时使用 `h' = m*h/(1-p)`，`m ~ Bernoulli(1-p)`；评估时不随机丢弃。它遮的是激活，不等于永久删除权重；高 dropout 也可能造成欠拟合。依据：[Dropout 原论文](https://www.jmlr.org/papers/v15/srivastava14a.html)。

## 跟随模型复习

MLP 学优化器与过拟合；[Autoencoder](../04_autoencoder/README.md) 学输入噪声与重构；[CNN](../05_cnn/README.md)/[ResNet](../06_resnet/README.md) 学增强、BatchNorm 与深层训练；[RNN](../08_rnn/README.md) 学 BPTT 和裁剪；[Transformer](../11_transformer/README.md)/[GPT](../12_gpt/README.md) 学 LayerNorm、AdamW、warmup 和序列 loss。Bias/归一化参数是否参与衰减必须显式记录，作为待比较配置；现有 GPT 对全部 `model.parameters` 统一衰减，并未实现参数分组或 scheduler。

计划产出：一张优化器对照表、train/validation 曲线、学习率曲线，以及每个技巧的独立消融和失败解释。
