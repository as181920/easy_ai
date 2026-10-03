# 04 · Auto-encoding：从压缩到去噪

状态：课程大纲，尚无训练实现。先修：01–03；基础全连接版本不需要 CNN。后续：[CNN](../05_cnn/README.md)、[生成模型](../13_generative/README.md)。

从分类 `x → y` 转为重构 `x → encoder → z → decoder → x_hat`。Encoder 学表示，Decoder 重建输入，瓶颈 z 控制容量；它与 Seq2seq 都有编码/解码结构，但任务目标和信息流不同。

| 顺序 | 最小实验 | 核心问题 |
| --- | --- | --- |
| 1. 线性 AE | 压缩带已知低维结构的合成向量，对照 PCA | 线性子空间、中心化、MSE；不能声称任意设置都与 PCA 参数一致 |
| 2. 非线性 bottleneck AE | 小 MLP 重构曲线/向量，逐渐改变 z 维数 | 容量、信息损失、泛化；没有瓶颈可能只学复制 |
| 3. Sparse AE | 给 latent 激活加稀疏约束 | 激活稀疏与权重稀疏不同；扫描强度避免表示全零 |
| 4. Denoising AE | 输入加噪或随机遮挡，target 保留干净输入 | corruption 是任务构造，与隐藏层 dropout 区分 |
| 5. 卷积 AE（学完 05） | 小图像压缩/去噪 | 复用卷积，比较形状与重构细节 |

MSE 适合连续重构；二元数据可使用相应 Bernoulli 重构目标，不能把任意像素值机械当作二元标签。比较 PCA、复制/均值基线；记录独立样本重构误差、latent 跨样本方差、重构前后图。只有训练重构漂亮不能证明有用表示，异常检测的重构误差也需另行验证。

输入 mask 必须明确遮挡率、填充值、是否传入可见性标记、loss 算全部位置还是仅遮挡位置；不同选择定义不同任务。去噪 AE 不一定只对遮挡位置算 loss。依据：[Denoising Autoencoder](https://www.jmlr.org/papers/v11/vincent10a.html)。

普通 AE 不保证随机抽取 z 能得到合理样本；VAE 的先验、KL 和重参数化放在 [13](../13_generative/README.md)，学会概率后再进入。BERT 的 masked-token prediction 和 MAE 的图像遮挡重构在 13 作为自监督扩展，不能把所有 Encoder/Decoder 都称为同一种 Autoencoder。

计划产出：压缩率/重构误差曲线、latent 图、干净/遮挡/重构对照，以及瓶颈、稀疏约束和 dropout 的独立消融。
