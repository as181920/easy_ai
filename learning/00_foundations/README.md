# 00 · 数学、数据与经典基线

已实现可运行的最小教学实验。先修：无。 下一章：[01_basic_nn](../01_basic_nn/README.md)。

先理解标量/向量/矩阵、shape、矩阵乘法和链式法则，再从平方误差手算一次更新。概率部分使用稳定的 sigmoid、softmax 和 log-sum-exp 交叉熵，避免直接对大 logits 求指数。

| 实现 | 数学与用途 | 实验 |
| --- | --- | --- |
| Linear | `ŷ=w·x+b`；MSE 梯度为 `2 Xᵀ(ŷ-y)/N` | 拟合 `y=2x+0.3` |
| Logistic | `p=sigmoid(w·x+b)`；BCE 梯度为 `Xᵀ(p-y)/N` | 无特征交互的线性基线拟合 XOR 象限 |
| PCA | 训练均值中心化 → 协方差 → 正交幂迭代 → 投影/重构 | 四维数据压成二维；零特征值使用正交补 |
| KMeans | 最近中心分配 → 簇均值更新 | 两团二维点；空簇保留上次中心 |
| CART / Forest / Boosting | 平方误差切分；bootstrap + 每节点特征抽样；残差拟合 | 和 Logistic 对照非线性分类 |

每个生成器用局部 Random，不消耗全局随机状态。训练/验证使用 seed 与 seed+1；标准化统计只能由训练集计算。学习 precision/recall 时先看混淆矩阵：precision=TP/(TP+FP)，recall=TP/(TP+FN)，分母为零时需明确约定。准确率不是类别不平衡任务的唯一指标。

```ruby
model = EasyAILearning::Foundations::Linear.new(features: 1)
model.fit([[0.0], [1.0]], [0.3, 2.3], steps: 60, lr: 0.1)
model.predict([0.25])
```

PCA 的线性子空间是后续 Autoencoder 的对照。树集成用于提醒：神经网络增加复杂度前也需要经典基线。只展示合成任务，不声称已经复现大型数据集基准。

## 数据、训练与独立推理

从仓库根目录运行；安装依赖用 `bundle install`。涉及训练与模型推理时默认 `auto`：优先 CUDA，不可用时回退 CPU；也可显式 `--device cpu`。使用小规模合成数据，无模型下载。限制线程能避免 tiny tensor 的 CPU 线程开销：

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/00_foundations/data.rb
bundle exec ruby learning/00_foundations/train.rb --steps 60
bundle exec ruby learning/00_foundations/predict.rb
bundle exec rake test:learning
```

`--seed` 改随机种子，`--output` 分开实验目录，训练还可用 `--device cpu/cuda/auto`。默认输出 `runs/learning/00_foundations/default/`，重跑会覆盖同名产物。`data.rb` 导出数据配方的样本用于查看；训练入口自行调用生成器，不依赖该 JSON 文件，训练中的特殊移位/遮挡在实验源码与实际 `data.json` 中记录。

推理 `--model PATH` 指向保存的模型。 `--value 0.25` 给线性回归输入。

## 结果与正确性

[实际运行记录](results.json) 保存 seed、步数、环境与指标；下图取自同一运行，不是验收阈值。

![本章实验结果](images/linear-loss.svg)

[实验代码](../lib/easy_ai_learning/foundations/experiment.rb) 串起各步骤；共享数据在 [course/data.rb](../lib/easy_ai_learning/course/data.rb)。核心验证见 [测试](../test/course/foundations_test.rb)，梯度对照另见 [derivatives_test.rb](../test/course/derivatives_test.rb)。测试只检查确定性公式、shape、mask、梯度、状态与参数更新；不训练到某个准确率或权重分布。

产物包括实际数据、JSON 推理状态、history、诊断与 SVG；本地完整参数/历史留在忽略的 runs 下，仓库只收录小型结果摘要与图。00/07 的非神经实验和 15 的交互训练输出格式按任务分别记录，不强行统一为分类 loss。
