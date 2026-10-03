# 04 · Auto-encoding：压缩、稀疏与去噪

已实现可运行的最小教学实验。先修：01–03。 下一章：[05_cnn](../05_cnn/README.md)。

复用 01 的 MLP 做 Encoder/Decoder，复用 02 的训练器与 03 的诊断。输入为 `x=[cos t,sin t,0.5cos t,0.5sin t]`，瓶颈二维，训练/验证由独立 t 样本生成。

`x → encoder → z → decoder → x̂`。线性版本是 `4→2→4`；非线性版本为 `4→12(tanh)→2→12(tanh)→4`。重构均方误差按所有样本与维度平均；sparse 版本另加 `λ mean(|z|)`，约束 latent 激活而非权重。Linear AE 有 22 参数，非线性 AE 有 174 参数。

实验递进：线性 AE/PCA 对照 → 非线性瓶颈 → latent L1 稀疏 → 30% 输入随机遮挡的 Denoising AE。输入遮挡使用 `x_corrupt=x*mask`，target 始终干净 x；本章 loss 计算所有位置，不是只计算遮挡位置。

```ruby
z = model.encode(x)
reconstruction = model.decoder.call(z)
loss = mse(reconstruction, clean_x) + sparsity * z.abs.mean
```

验证去噪使用固定 mask，训练每步生成新 mask。输出重构明细、loss 曲线和 latent 统计；稀疏惩罚可能损害重构，训练重构好也不证明表示有用。当前数据有精确二维线性结构，所以 PCA 可接近零误差；非线性网络不必在短训练内更好。

普通 AE 不能保证随机 latent 样本合理；13 的 VAE 引入先验/KL。卷积 AE 在 05 复用本章目标，MAE 在 13 将遮挡移到 patch。依据：[Denoising AE](https://www.jmlr.org/papers/v11/vincent10a.html)。

## 数据、训练与独立推理

从仓库根目录运行；安装依赖用 `bundle install`。涉及训练与模型推理时默认 `auto`：优先 CUDA，不可用时回退 CPU；也可显式 `--device cpu`。使用小规模合成数据，无模型下载。限制线程能避免 tiny tensor 的 CPU 线程开销：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/04_autoencoder/data.rb
bundle exec ruby learning/04_autoencoder/train.rb --steps 60
bundle exec ruby learning/04_autoencoder/predict.rb
bundle exec rake test:learning
```

`--seed` 改随机种子，`--output` 分开实验目录，训练还可用 `--device cpu/cuda/auto`。默认输出 `runs/learning/04_autoencoder/default/`，重跑会覆盖同名产物。`data.rb` 导出数据配方的样本用于查看；训练入口自行调用生成器，不依赖该 JSON 文件，训练中的特殊移位/遮挡在实验源码与实际 `data.json` 中记录。

推理 `--model PATH` 指向保存的模型。 `--input PATH` 可指定 `{"input": ...}` JSON；默认提供一个符合本章形状的示例。Seq2seq/EncoderDecoder 输出自由生成，GPT 输出 top-1 生成，其他模型输出 score/reconstruction；RL actor-critic 输出 policy logits 与 value。

## 结果与正确性

[实际运行记录](results.json) 保存 seed、步数、环境与指标；下图取自同一运行，不是验收阈值。

![本章实验结果](images/bottleneck-loss.svg)

[实验代码](../lib/easy_ai_learning/autoencoder/experiment.rb) 串起各步骤；共享数据在 [course/data.rb](../lib/easy_ai_learning/course/data.rb)。核心验证见 [测试](../test/course/vision_test.rb)，梯度对照另见 [derivatives_test.rb](../test/course/derivatives_test.rb)。测试只检查确定性公式、shape、mask、梯度、状态与参数更新；不训练到某个准确率或权重分布。

产物包括实际数据、JSON 推理状态、history、诊断与 SVG；本地完整参数/历史留在忽略的 runs 下，仓库只收录小型结果摘要与图。00/07 的非神经实验和 15 的交互训练输出格式按任务分别记录，不强行统一为分类 loss。
