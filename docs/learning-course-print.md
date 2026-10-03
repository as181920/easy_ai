# AI 学习课程 · 精简打印版

阅读顺序：数学/数据 → MLP → 训练/诊断 → AE/CNN/ResNet → Tokenizer/RNN/Seq2seq → Attention/Transformer/GPT → 生成/迁移/RL → 综合实验。先学 00–03，再选择分支；RL 不以 GPT 为先修。代码片段保留计算核心，省略 import、配置、日志与文件读写；完整实现见 `learning/lib/easy_ai_learning/`，各章入口与结果见 `learning/<编号>_<主题>/`。

## 00 · 数学、概率、数据与经典基线

张量形状先于公式：Linear 对最后一维做投影；batch 维是样本，参数在样本间共享。矩阵乘法 `[B,D]@[D,O]→[B,O]`。参数参与梯度更新，超参数决定结构或训练规则。

线性回归：`ŷ=Xw+b`，`L=mean((ŷ-y)²)`，`∂L/∂w=2Xᵀ(ŷ-y)/N`。逻辑回归：`p=sigmoid(Xw+b)`，BCE 梯度 `Xᵀ(p-y)/N`。多类分类使用 logits 与交叉熵，不能把原始 score 当作概率。

```ruby
# 大 logits 下稳定的 softmax 和交叉熵
largest = logits.max
exp = logits.map { |v| Math.exp(v - largest) }
probabilities = exp.map { |v| v / exp.sum }
loss = largest + Math.log(exp.sum) - logits[target]
```

链式法则：`∂L/∂x=(∂L/∂h)(∂h/∂x)`；用 `f'(x)≈[f(x+ε)-f(x-ε)]/(2ε)` 检查梯度。训练/验证/测试分开；标准化只在训练集估计均值/标准差。准确率=正确数/N；precision=TP/(TP+FP)，recall=TP/(TP+FN)。分母为零须明确约定。

PCA：训练均值中心化 → 协方差特征向量 → 低维投影/重构。KMeans：最近中心分配 → 更新簇均值。CART 按误差选择切分；Forest 叠加 bootstrap 与随机特征；Boosting 逐步拟合当前残差。它们是增加神经复杂度前的基线。

源码：`foundations/{math,linear,pca,k_means,tree,ensemble}.rb`。

## 01 · 最小 MLP：9 参数 XOR

单个线性阈值无法分开 XOR；引入两个 ReLU 单元即可。输入/目标为 `[0,0]→0`、`[0,1]→1`、`[1,0]→1`、`[1,1]→0`。`2→2→1` 网络参数量 `(2×2+2)+(2×1+1)=9`。

```ruby
# 精确构造用于证明存在解，不用于初始化训练
h1 = [x1 + x2, 0].max
h2 = [x1 + x2 - 1, 0].max
score = h1 - 2 * h2
prediction = score >= 0.5 ? 1 : 0
```

一般模型：`z=W1x+b1`，`h=ReLU(z)`，`ŷ=W2h+b2`。MSE 反向：`dŷ=2(ŷ-y)/N`，`dW2=dŷ·hᵀ`，`dz=(W2ᵀdŷ)⊙1[z>0]`，`dW1=dz·xᵀ`。全四点拟合正确不等于泛化；中间点并没有新的 XOR 真值标签。Score 可超出 `[0,1]`。

```ruby
class Mlp < Torch::NN::Module
  def initialize
    super()
    @hidden = Torch::NN::Linear.new(2, 2)
    @output = Torch::NN::Linear.new(2, 1)
  end
  def forward(x)
    @output.call(Torch.relu(@hidden.call(x)))
  end
end
```

源码：`basic_nn/{logic_network,scalar_logic_network,mlp}.rb`。

## 02 · 训练：SGD、Momentum、Adam、AdamW

训练循环：forward → loss → zero_grad → backward → 诊断/裁剪 → step → eval/no_grad 验证。明确 step、batch、epoch 和 loss 的平均方式。课程训练默认 `auto`，优先 CUDA、不可用时 CPU；核心测试用 CPU 检查确定性逻辑。

```ruby
optimizer.zero_grad
loss = objective.call
loss.backward
clip_gradients(model.parameters, max_norm) # 如有需要
optimizer.step
```

SGD：`θ←θ-ηg`。Momentum：`v←μv+g`，`θ←θ-ηv`。Adam：`m←β1m+(1-β1)g`，`v←β2v+(1-β2)g²`，修正偏差后除以 `sqrt(v̂)+ε`。AdamW 将衰减与梯度矩估计分开；Adam 加 L2 一般不等价于 AdamW。

```ruby
# AdamW 的核心；t 为该参数真实更新次数
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient.square
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
parameter.mul!(1 - lr * decay)
parameter.add!(m_hat / (v_hat.sqrt + epsilon), alpha: -lr)
```

学习率太小可能不动，太大可能发散；warmup 后可按 schedule 衰减。Xavier/He 初始化关注初始信号尺度；不要将全部隐藏权重设成相同值。输入标准化、weight decay、dropout、early stopping、数据增强分别消融，不同时添加后强行归因。

Dropout 训练为 `h'=m⊙h/(1-p)`，`m~Bernoulli(1-p)`，评估不丢弃；它随机遮激活，不永久删除权重。梯度裁剪按全局范数缩放，不能代替检查爆炸原因。累积 microbatch mean loss 时按真实样本/token 数加权。Loss scaling 先放大 loss，step 前反缩放梯度；非有限梯度跳过更新。显式 scaler 不等于完整 AMP/autocast。

比较优化器需控制数据、结构、预算并给各自公平学习率搜索。推理权重不包含优化器矩；恢复还需 step/schedule、随机与数据状态。最佳验证模型与最后 optimizer 不一定属于同一时刻。

源码：`training/{scalar_optimizer,optimizer,loop,math,accumulation,early_stopping,grad_scaler}.rb`。

## 03 · 权重、激活、梯度与更新诊断

**没有通用的合理权重直方图。集中于零附近不自动代表不好，分散也不自动代表好。** 稀疏任务、衰减、初始化、归一化与等价的层间缩放都能改变权重形状。不同层/bias/norm scale 分开看。

| 对象 | 核心观测 | 需要确认的现象 |
| --- | --- | --- |
| 权重 | 均值、std、RMS、分位数、极值、近零比例 | 突变、异常值、整体尺度 |
| 激活 | 零比例、饱和、跨样本方差 | dead units、表示坍缩 |
| 梯度 | 缺失/零/非有限、每层范数、裁剪前后 | 冻结、断图、消失、爆炸 |
| 实际更新 | 绝对范数、相对范数 | 是否在学、更新过小/过剧烈 |
| 泛化 | 验证指标、错误样本、输出分布 | 记忆、常量预测、泄漏 |

```ruby
absolute = (after - before).norm.item
relative = absolute / (before.norm.item + 1e-12)
rms = Math.sqrt(values.sum { |v| v * v } / values.size)
```

近零参数的 relative update 易失真，还需看绝对值。AdamW 实际更新不是简单 ηg。奇异值归一化为 p 后，`effective_rank=exp(-Σp log p)`；低秩是否有害取决于任务。初始化参考：Xavier `Var≈2/(fan_in+fan_out)`；ReLU 的 He `Var≈2/fan_in`，不是训练后的验收目标。

不更新先查注册/冻结/图/标签；发散先查输入、数值操作和步长，再比较初始化、归一化、残差与裁剪。训练好验证差再考虑数据、容量和正则化。写出“现象→证据→假设→单变量干预→独立验证”。

源码：`diagnostics/stats.rb`。

## 04 · Autoencoder：压缩、稀疏与去噪

`x→encoder→z→decoder→x̂`，目标是重构。线性 AE 对照 PCA；非线性 AE 引入 MLP；瓶颈限制信息容量；Sparse AE 约束 latent 激活，不等于权重稀疏。

```ruby
latent = encoder.call(input)
reconstruction = decoder.call(latent)
loss = (reconstruction - clean_target).square.mean
loss = loss + sparsity * latent.abs.mean
```

去噪：输入加噪/遮挡，target 保持干净；隐藏层 dropout 与输入 corruption 的语义不同。定义遮挡率、填充值、是否提供可见标记、loss 在全部还是仅遮挡位置计算。本章 denoising loss 计算全部位置。

重构好不证明表示有用；无瓶颈可能只是复制；普通 AE 不保证随机 z 能生成合理样本。先看独立重构，再看下游表示/采样。卷积 AE 在 05，概率 latent 的 VAE 在 13。

源码：`autoencoder/{model,convolutional}.rb`。

## 05 · CNN：局部连接、共享参数与池化

深度学习常用 cross-correlation；核不翻转。参数在空间位置共享。输出尺寸 `floor((I+2P-K)/S)+1`；参数量 `C_out*C_in*K_h*K_w+C_out`。通道、stride、padding、感受野逐层计算。

```ruby
features = Torch.relu(conv1.call(images))
features = Torch::NN::Functional.max_pool2d(features, 2)
features = Torch.relu(conv2.call(features))
logits = head.call(features.mean([2, 3])) # global average pooling
```

池化压缩空间；global average pooling 不等于 flatten。BatchNorm 训练更新 running mean/variance，评估使用保存统计；buffer 也需要随模型保存。数据增强只用于训练，并保证不改变标签：横/竖线图做 90° 旋转会交换类别。

卷积 AE 用卷积编码、下采样、上采样和卷积解码重构图像。参数少或训练准确率高都不能单独证明泛化更好。

源码：`cnn/{convolution,model}.rb`、`autoencoder/convolutional.rb`。

## 06 · ResNet：学习残差

`y=x+F(x)` 给深层网络一条直接信息/梯度路径；shape 不同则 projection shortcut。原 basic block 相加后还有 ReLU；pre-activation 最后直接相加，零 residual 能保留负值。

```ruby
residual = norm2.call(conv2.call(Torch.relu(norm1.call(conv1.call(x)))))
y = Torch.relu(shortcut.call(x) + residual)
```

Bottleneck 用 `1×1→3×3→1×1` 缩小中间通道。比较普通深层 CNN 与同深度残差 CNN 时控制数据、初始权重、归一化和预算。残差不能保证梯度永远稳定或每次验证提升。11 复用该思想；归一化类型改为 LayerNorm。

源码：`resnet/{block,model,bottleneck,preactivation}.rb`。

## 07 · Tokenizer、BPE 与 Embedding

字符分词先建训练词表，未见字符需要 UNK；UTF-8 byte 分词固定 256 基础符号，能覆盖未见文本；BPE 从高频相邻对建立合并规则。完整保存词表/规则，编码与解码必须使用同一规则。

```ruby
counts = tokens.each_cons(2).tally
pair = counts.max_by { |_, frequency| frequency }.first
# 从左到右不重叠地将 pair 换成新 ID；按学习顺序应用规则
vectors = embedding.call(token_ids) # [B,T] → [B,T,D]
```

Tokenizer 规则学习与 Embedding 梯度更新不同。重复查同一 ID 时梯度累加；未查的行无该任务梯度。训练词表后再编码验证/测试，避免泄漏。比较词表大小、序列长度、OOV 和可逆性；不能仅凭 token 少判断好坏。历史按词切分版本可能归一化空白；本轮可逆 byte BPE 保留全部字节。

源码：`tokenizers/{character,reversible_bpe}.rb`。

## 08 · RNN、LSTM、GRU 与 BPTT

RNN：`h_t=tanh(Wx_t+Uh_(t-1)+b)`。时间展开但参数共享，长度增长不增加共享参数量；BPTT 沿展开图反传。长记忆可能梯度消失/爆炸，裁剪只控制爆炸幅度。

```ruby
# LSTM gates 已由 input/recurrent Linear 计算
cell = sigmoid(forget) * previous_cell + sigmoid(input_gate) * tanh(candidate)
hidden = sigmoid(output_gate) * tanh(cell)
# GRU reset-after 变体
candidate = tanh(input_candidate + reset * recurrent_candidate)
hidden = (1 - update) * candidate + update * previous_hidden
```

LSTM 显式保留 cell state；GRU 用 reset/update 门。Reset 放在 recurrent projection 前/后有不同公式，必须说明。独立序列重置 state；连续流才传入历史 state。截断 BPTT 用 detach 停止跨边界梯度。

Padding loss mask 只控制计分，不自动阻止状态更新；有效长度之外需保留原 state。实验从短 next-token 到延迟复制，再比较训练长度与更长序列，不能以某 seed 的收敛作为单元测试。

源码：`rnn/{cell,model}.rb`。

## 09 · Seq2seq 与 teacher forcing

Encoder 读输入，将状态交给 Decoder；训练/推理信息流分开。反转 `[a,b,c,d]` 的 target 是 `[d,c,b,a,EOS]`，decoder input 是 `[BOS,d,c,b,a]`。

```ruby
state, memory = encode_memory(source)
state, logits = decode_step(previous_token, state, memory)
# 训练 previous_token 来自右移真值；推理来自上一轮预测
```

Teacher forcing 不能输入同位置答案；生成时误差可连锁传播，所以同时看 teacher-forced token accuracy、自由生成和整序列正确率。定义 EOS/最大长度；已结束样本不继续输出随意符号。固定状态瓶颈在 10 用检索 encoder states 改进。

源码：`seq2seq/model.rb`。

## 10 · Attention 与四种 mask

Additive attention 先理解 query 检索 memory；scaled dot-product 使用 `softmax(QKᵀ/sqrt(d_k))V`。Self-attention 同一序列作 Q/K/V，cross-attention 的 query 与 memory 来源不同；多头分组特征后再拼接。

```ruby
scores = Torch.matmul(q, k.transpose(-2, -1)) / Math.sqrt(head_dim)
scores = scores.masked_fill(valid.logical_not, -Float::INFINITY)
weights = scores.softmax(-1)
context = Torch.matmul(weights, v)
```

| Mask | 核心语义 |
| --- | --- |
| Dropout | 随机丢激活/attention 权重，评估关闭 |
| Input corruption | 改输入，定义去噪/遮挡预测目标 |
| Attention padding/causal | softmax 前决定 key 可见性；causal 禁止看未来 |
| Loss | 决定哪些 target 计分，按有效位置数归约 |

全遮挡 query 要显式处理，避免 softmax 全 -Inf。Key padding mask 不自动处理无效 query/target。Softmax 后 dropout 的单次行和不必为 1；图要说明观测时点。Attention 权重不是模型参数，也不自动是因果解释。

源码：`attention/{additive,multi_head,causal_self_attention}.rb`。

## 11 · Transformer：位置、残差、LayerNorm 与 FFN

Embedding 加位置，attention 混合 token，FFN 逐位置共享。Encoder 双向；Decoder 因果；EncoderDecoder 额外 cross-attention。无位置的双向 encoder 是排列等变的，不能凭空知道顺序。

```ruby
x = token_embedding.call(ids) + positional_embedding.call(ids)
x = x + attention.call(layer_norm1.call(x))
x = x + feed_forward.call(layer_norm2.call(x))
```

Pre-LN 先归一化再计算支路，Post-LN 在残差相加后归一化。LayerNorm 对每个位置的特征维统计，BatchNorm 跨样本/空间并维护 running stats；不能混称权重归一化。Sinusoidal 的第 2i/2i+1 维使用 `sin/cos(pos/10000^(2i/D))`。对位置、残差、归一化逐项消融，浅层小任务上的表现不能直接推广到大模型。

源码：`transformer/{encoder_block,block,feed_forward,sequence_model,encoder_decoder,sinusoidal}.rb`。

## 12 · GPT：因果 next-token 与采样

Decoder-only causal Transformer，预测当前上下文的下一 token。输入 `[a,b,c]` 对应 target `[b,c,d]`；训练可并行计算各位置，生成必须逐步追加。

```ruby
loss = cross_entropy(logits.reshape([-1, vocab_size]), targets.reshape([-1]))
context_logits = model.call(context)
last_logits = context_logits.narrow(1, context_logits.shape[1] - 1, 1).squeeze(1)
probabilities = (last_logits / temperature).softmax(-1)
next_token = Torch.multinomial(probabilities, num_samples: 1)
```

Temperature 必须正；top-k 只保留最大的 k 个 logits；超出 context 时截取最后窗口。Generate 用 eval/no_grad，结束后恢复原训练模式。

Perplexity=`exp(mean token cross-entropy)`，不同 tokenizer/计分方式不可直接比较。训练 loss 降或文本像语料不代表语义与事实正确。固定六符号循环可验证因果结构，但只有少数模式，不能称为自然语言泛化。

源码：`gpt/{model,experiment,batcher,text_dataset}.rb`。

## 13 · 概率生成、对抗、扩散与遮挡重构

普通 AE 重构不约束随机 latent 的分布；VAE 给 latent 先验与 KL。Encoder 预测 μ/log variance，重参数化保持可微：

```ruby
z = mean + (0.5 * log_variance).exp * noise
kl = -0.5 * (1 + log_variance - mean.square - log_variance.exp).sum(1).mean
loss = (decoder.call(z) - x).square.sum(1).mean + beta * kl
```

β=1 对应原始目标的 KL 权重；本轮 β=0.1 是教学变体。分别观察重构、KL、posterior collapse 与先验采样。重构项按维求和，不能与 AE 的 mean MSE 数值直接相比。

GAN：D 区分 real 与 `G(z).detach`，G 用 non-saturating `BCE(D(G(z)),1)`。D 的 backward 不进入 G；G 的 backward 要经过 D。双方 loss 不直接等价于质量，观察覆盖、多样性和 mode collapse。

DDPM：`α_t=1-β_t`，`ᾱ_t=Π α_s`；训练预测噪声：

```ruby
noisy = cumulative_alpha.sqrt * x + (1 - cumulative_alpha).sqrt * noise
loss = (predict_noise(noisy, time) - noise).square.mean
# 反向均值：(x_t - β_t/sqrt(1-ᾱ_t)*ε_θ)/sqrt(α_t)
# posterior variance：β_t*(1-ᾱ_(t-1))/(1-ᾱ_t)；最后一步为零
```

Masked language 从双向上下文预测遮挡 token，loss 只计遮挡位置；不能冒充因果生成。MAE 将图像分 patch，encoder 只读可见 patch，decoder 用 mask token 恢复位置并重构不可见 patch。表示重构、随机生成和数据分布是不同目标。

源码：`generative/{vae,gan,diffusion,masked_autoencoder}.rb`。

## 14 · 迁移、冻结、LoRA 与蒸馏

先训源任务，再比较目标任务的从头训练、冻结特征、部分解冻、全量微调与低秩适配。冻结参数要从 optimizer 更新中排除；BatchNorm buffer 更新与梯度冻结不同。迁移可能负迁移。

```ruby
# LoRA：base 冻结；A 随机、B=0，初始行为与 base 相同
output = base.call(x) + (alpha / rank) * b.call(a.call(x))
merged_weight = base.weight + (alpha / rank) * Torch.matmul(b.weight, a.weight)
```

LoRA 限制更新秩，不要求 base 权重稀疏。保存 base+A+B；比较合并/未合并 forward。蒸馏将 teacher 软目标 detach，student 在温度 T 下计算 soft CE，乘 T² 后和真实标签 CE 混合：

```ruby
soft_target = (teacher_logits.detach / temperature).softmax(-1)
soft_loss = -(soft_target * (student_logits / temperature).log_softmax(-1)).sum(-1).mean
loss = soft_weight * temperature**2 * soft_loss + (1 - soft_weight) * hard_loss
```

Soft CE 与 KL 差 teacher entropy 常数，gradient 等价但数值不同。Teacher 错误也可能传给 student；同时观察目标泛化、参数量、实际更新与遗忘。

源码：`transfer/{low_rank_linear,adapted_mlp,distillation}.rb`。

## 15 · RL：Bandit、Q-learning、DQN、策略梯度与 PPO

监督学习用标签；RL 从互动奖励学习期望回报。MDP 包含 state/action/transition/reward；Markov 性要求状态包含预测下一步需要的信息。Bandit 是单步起点；多步任务要解决延迟信用分配。

`G_t=r_t+γr_(t+1)+…`。Bandit 用探索与样本均值；exact Bellman reference 已知环境；Q-learning 从实际 transition 更新。Terminated 与 time-limit truncated 分开：终止不 bootstrap，continuing-task 的时间截断可 bootstrap。

```ruby
target = reward + (terminated ? 0 : gamma * q[next_state].max)
q[state][action] += alpha * (target - q[state][action])
# DQN 用 replay、detached target Q、定期同步的 target network
```

REINFORCE：`L=-mean(logπ(a|s)*G)`；baseline 不应依赖本次动作。Actor-critic 学 value 并用 advantage。GAE：`δ_t=r_t+γ(1-terminal)V(next)-V(s)`，反向 `A_t=δ_t+γλ(1-terminal)A_(t+1)`；不能跨 episode 混回报。

```ruby
ratio = (new_log_probability - old_log_probability.detach).exp
surrogate = Torch.minimum(ratio * advantage.detach,
  ratio.clamp(1 - clip, 1 + clip) * advantage.detach)
policy_loss = -surrogate.mean
```

PPO 固定一次 rollout 的 old logp，允许有限次数重复更新；clip 根据 advantage 正负取 pessimistic objective，不等于直接把 loss 截断。Value target detach，policy/critic 分开观察。评估关闭更新，报告 return、成功率和路径；critic loss 降不证明策略好。确定性环境的不同种子不创造环境波动。

源码：`rl/{chain,tabular,replay,objectives,actor_critic}.rb`。

## 16 · 综合实验与正确性证据

完整闭环：数据来源/划分 → 经典基线 → shape/参数量 → 固定预算训练 → 验证选型 → 独立 test → 诊断/失败解释 → 保存与独立推理。使用多训练 seed 报 mean/std，不根据 test 反复调参。简单任务全部满分可能只说明任务太容易。

核心逻辑用确定性测试：闭式/手算公式、有限差分/autograd、shape、卷积/门控、mask/未来泄漏、冻结/梯度隔离、参数一步更新、保存加载。**不将训练准确率、收敛步数、某种 weights 分布或 loss 必须下降写成单元测试条件。**

```ruby
numerical = (loss_at(value + epsilon) - loss_at(value - epsilon)) / (2 * epsilon)
# 对照 analytic gradient；epsilon/tolerance 结合 dtype 和数值尺度选择
expected = old_parameter - learning_rate * gradient # SGD 一步精确参考
```

相同数据与 eval/no_grad 下，加载前后 logits 应一致；恢复训练还需同一时刻的模型/优化器与随机/数据状态。枚举值如 `kind`、`activation`、`device` 兼容 string/symbol，入口统一规范化并拒绝未知选项。

源码：`capstone/experiment.rb`；核心测试 `learning/test/course/`，全部实验与完整说明 `learning/README.md`。

理论原文：Xavier（Glorot/Bengio, 2010）、He 初始化/ResNet（He 等, 2015）、Dropout（Srivastava 等, 2014）、AdamW（Loshchilov/Hutter, 2017）、Attention Is All You Need（Vaswani 等, 2017）、VAE（Kingma/Welling, 2013）、DDPM（Ho 等, 2020）、PPO（Schulman 等, 2017）；各章 README 提供链接。
