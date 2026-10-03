# 13 · 生成模型与自监督重构

状态：课程大纲，尚无训练实现。先修：04 和 00 的概率；图像分支加 05，文本/Transformer 分支加 11–12。本章各主题可选择，不是一次实验同时实现。

| 顺序 / 分支 | 学习重点 | 最小实验与诊断 |
| --- | --- | --- |
| VAE（接 AE） | latent 概率、先验、重参数化、ELBO 与 KL | 简单二维/图像数据，对照重构、随机采样、KL 与 posterior collapse |
| GAN | generator/discriminator、交替训练 | 低维分布拟合，观察 mode collapse 与双方失衡 |
| Diffusion | 前向加噪、噪声预测、反向采样 | 小二维分布，再扩展小图像；区分训练与采样步数 |
| BERT 遮挡预测（语言选修） | 双向上下文、masked-token objective | 比较 GPT next-token；不能用其双向输入冒充因果生成 |
| MAE（视觉选修） | patch 遮挡、可见输入与重构位置 | 复习 04 的 input/loss mask，结合 ViT/Transformer |

普通 AE 的低重构误差不保证随机 latent 样本合理；VAE 对 latent 分布增加约束。来源：[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)。原论文目标是 ELBO；KL 权重扫描等扩展需与基本版本区分。

BERT/MAE 是自监督表示学习扩展，不统一当作自回归生成器；它们与去噪 AE 的共同点是从不完整输入预测信息。不同架构的 mask 可见性和计分位置需要逐项定义。

计划产出：重构与生成分开展示，独立数据评估、样本覆盖/多样性与失败案例。模型参数分布、latent 分布和输出数据分布必须分别观察，不能混为一个“权重是否合理”的问题。
