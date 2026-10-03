# 06 · ResNet：让深层网络学残差

状态：课程大纲，尚无实现。先修：05。残差原理在 [Transformer](../11_transformer/README.md) 再复用。

ResNet 应纳入基础课程：它把「网络更深为什么可能更难训练」与可复用的残差连接联系起来。先学 `y = x + F(x)`，再学卷积残差块；shape 不同时使用 stride/projection shortcut，不能直接相加。原论文：[Deep Residual Learning](https://arxiv.org/abs/1512.03385)。

学习顺序：同宽普通深层网络 → identity shortcut → basic block → BatchNorm 与训练/评估模式 → projection → bottleneck → pre-activation 选修。原始块和 pre-activation 的排列不同，逐图标出激活所在位置。

计划实验在同一小图像数据上比较浅 CNN、加深普通 CNN、同深度残差 CNN，记录训练误差、验证误差、逐层激活/梯度和计算成本。比较需说明参数量、预算、初始化、optimizer、增强和多种子；残差降低优化困难不保证每个小实验都提高验证准确率。

训练技巧先沿用 02，逐项对照初始化、学习率、BatchNorm、SGD+Momentum/AdamW。不要同时添加全部技巧后把改善归因于残差。消融 identity/projection、去掉归一化等必须保证 shape 与训练设置可比较。

完成标准：推导 identity shortcut 提供的直接梯度路径，理解它不能保证梯度永不消失/爆炸；能联系 Transformer 的残差结构，同时区分 BatchNorm 与 LayerNorm。
