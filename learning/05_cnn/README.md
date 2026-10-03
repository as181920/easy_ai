# 05 · CNN：局部连接与共享参数

状态：课程大纲，尚无实现。先修：01–03。下一章：[ResNet](../06_resnet/README.md)。

1. 手算一维/二维卷积，区分数学卷积与深度学习常用 cross-correlation；理解 kernel、channel、stride、padding 和输出 shape。
2. 局部连接与参数共享；计算 `C_out * C_in * k_h * k_w + C_out` 参数，解释感受野。
3. Pooling、flatten、global average pooling；从单卷积到 LeNet 风格分类器。
4. 训练/验证：图像缩放与标准化、数据增强、BatchNorm 的训练/评估统计；比较 SGD+Momentum 与 AdamW。
5. AlexNet/VGG 作为结构阅读与深度/参数量对照，不要求初学者复现大数据训练。

计划实验先使用可生成的小形状图像，再选 MNIST 等小公开数据。比较 MLP、CNN 和简单基线，记录数据来源、分组划分、train/validation 指标、混淆矩阵、特征图。增强只用于训练；标签必须在所选变换下有效。

完成标准：能手算每层 shape/参数量、解释平移性质与边界效应，识别卷积层的激活与梯度异常；不要把参数更少或训练准确率更高直接当作泛化更好。卷积 AE 回到 [04](../04_autoencoder/README.md) 扩展。
