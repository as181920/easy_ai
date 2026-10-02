# Basic neural networks: logic gates

## Three ways to calculate a gate

1. **Explicit branches:** `LogicGates.call` uses `if/else` to compute AND, OR, NAND and XOR. These results define the targets; branch logic is never called by the neural forward calculation.
2. **Single perceptron:** fixed affine scores followed by a threshold implement AND (`x1+x2-1.5`), OR (`x1+x2-0.5`) and NAND (`-x1-x2+1.5`). XOR is not linearly separable.
3. **Multilayer network:** two ReLU hidden units followed by four linear outputs can calculate all four gates simultaneously. Stacking affine layers without a nonlinear activation would still produce an affine function.

```text
x1 ----+
       +--> Dense(2,2) --> ReLU --> Dense(2,4) --> AND / OR / NAND / XOR
x2 ----+

hidden: 2*2 weights + 2 biases =  6
output: 2*4 weights + 4 biases = 12
                              ----
                               18 independently trainable parameters
```

Two hidden neurons suffice and one cannot express XOR with a linear output. This count assumes ordinary dense connections and trainable biases. Fixed weights, sharing or skip connections change the parameter count; 18 is not a universal minimum across all possible programs.

An exact mathematical construction proves that a solution exists:

```text
h1 = ReLU(x1 + x2)
h2 = ReLU(x1 + x2 - 1)
AND = h2
OR = h1 - h2
NAND = 1 - h2
XOR = h1 - 2*h2
```

Run the fixed demonstrations:

```bash
bundle exec ruby learning/01_basic_nn/logic.rb
```

## Learn the coefficients

```bash
bundle exec ruby learning/01_basic_nn/train.rb
# Default: CUDA if available; force CPU with --device cpu.
# Separate experiment:
bundle exec ruby learning/01_basic_nn/train.rb \
  --seed 2027 --steps 10000 --learning-rate 0.1 \
  --output runs/learning/basic_nn/logic-gates-2027
```

The trained model never copies the exact construction. Initial hidden weights are seeded random positive values; one bias starts near zero and the other negative, helping avoid initially dead ReLU neurons. Output coefficients are random too. Every one of the 18 coefficients is updated. This is a deliberate initialization policy, not a guarantee that every seed converges.

For input `x`, hidden preactivation `z = W1*x+b1`, activation `h = max(0,z)`, and output `y = W2*h+b2`:

```text
L = sum((y-target)^2) / 16       # four inputs times four output gates

dy = 2*(y-target)/16
dW2 += dy outer h;  db2 += dy
dz = (W2^T * dy) * (z > 0)      # derivative chosen as zero at z=0
dW1 += dz outer x;  db1 += dz

parameter -= learning_rate * accumulated_gradient
```

All gradients use the old weights; updates occur after processing all four examples. The runnable model uses `Torch::NN::Linear`, ReLU, mean MSE, `loss.backward`, and `Torch::Optim::SGD`; all tensor calculations run on the selected device. The separate `ScalarLogicNetwork` explicitly implements the chain rule above as a teaching reference. No Boolean shortcut computes the learned output. The complete input domain is only four rows, so the full truth table is the training set. The trainer stops when every output is within 0.01 of its target, or reports unsuccessful training at the step limit. Thresholding scores at 0.5 yields Boolean predictions; raw scores can slightly exceed `[0,1]` and are not probabilities.

## Observed runs and plots

![Learned logic gate decision boundaries, ReLU and trained XOR regions](images/logic-functions.png)

This snapshot uses the CUDA-trained coefficients, not the fixed exact construction. Panels are ordered **AND, OR, NAND, XOR, ReLU, then the trained model's XOR regions with all four inputs**.

The first four panels are decision boundaries in the `(x1, x2)` plane: black contour `trained_score(x1,x2) = 0.5`, light blue predicted 0, light orange predicted 1. Dots and labels show Boolean targets; axes run from 0 to 1.2. Only the four Boolean corners were trained.

The final panel evaluates the saved model on a dense 2D input grid, with **0.01 increments on both inputs** (121 × 121 points over `[0,1.2]`). Green shading is `model_xor_score >= 0.5`, blue shading is `< 0.5`; black contours come from the trained model's score 0.5. `(0,1)` and `(1,0)` are both in the same green positive band; `(0,0)` and `(1,1)` are in the blue negative regions outside it. This is the trained model's input-plane decision map, not hand-written reference lines. ReLU makes its actual contour piecewise linear, so two boundary components are expected for this learned solution.

The fresh seed-20261002 run begins with unequal input weights and ends with nearly equal, independently trained values. Their final `x1_weight - x2_weight` differences are `-1.1920928955078125e-07` and `-4.887580871582031e-06`; unlike the earlier seed-1337 run, neither pair is exactly equal. All four gates have targets invariant under swapping `x1` and `x2`, and full-batch training rewards symmetric outputs. The implementation does not share, average or copy the two input weights. Symmetric output functions need not always have symmetric internal parameters; this observed convergence depends on the architecture and initialization too. Small differences are invisible at PNG resolution, so the boundary still looks straight and parallel. Binary shading shows only the thresholded class, not the continuously varying raw score.

The plotting code loads the full JSON coefficients and exports contour coordinates with 17 significant digits; the grid spacing is 0.01 and contour extraction interpolates sampled scores. It is a numerical visualization, not an exact analytic contour. Original Float32 training precision is preserved; decimal rounding is not applied to model coefficients or scores before plotting.

A prior 1D view fixed `x2=0`; that omitted `(0,1)` and could not demonstrate the full XOR separation. The final panel now varies **both** coordinates. The optional saved `xor-curve.dat` still provides the 1D output-function slice for comparison; it is not the README's final panel. Loss curves remain in terminal output and saved `plots.txt` / `loss.json`.


For comparison, fixed perceptron boundaries are:

```text
AND:   x1 + x2 - 1.5 = 0     output 1 on the positive side
OR:    x1 + x2 - 0.5 = 0     output 1 on the positive side
NAND: -x1 - x2 + 1.5 = 0     output 1 on the positive side
```

The exact ReLU XOR construction has two boundaries, `x1+x2=0.5` and `x1+x2=1.5`; its positive region is the band between them. XOR cannot use a single affine decision boundary. The displayed contours are sampled from the saved trained model, so they are not replaced by these hand-written reference lines. Fitting the four corners does not uniquely determine the boundary between them.


Displayed snapshot: a fresh CUDA-trained **seed 20261002** model from `runs/learning/basic_nn/logic-gates/model.json`: **1,271 updates**, MSE `2.2769352653995156e-05`, maximum error `0.00998555589467287`, all 16 Boolean outputs correct. All model-derived panels use this same saved model. The prior seed-1337 run is preserved in ignored `runs/learning/basic_nn/logic-gates-1337-before-retrain/`. Initial coefficients for this fresh run are saved in `runs/learning/basic_nn/logic-gates/initial_parameters.json`.

| Seed | Updates to max-error <= 0.01 | Final MSE | Maximum error | Correct Boolean outputs |
| --- | ---: | ---: | ---: | ---: |
| 1337 (earlier default) | 1,623 | 1.515922e-5 | 0.009951 | 16/16 |
| 2027 | 1,293 | 2.722712e-5 | 0.009997 | 16/16 |
| 3407 | 1,801 | 1.532161e-5 | 0.009985 | 16/16 |
| 20261002 (displayed) | 1,271 | 2.2769352653995156e-05 | 0.00998555589467287 | 16/16 |

Different seeds change initial coefficients, the trajectory and final parameters; the required truth table remains the same. The script deliberately defaults to seed 1337 for reproducibility: rerunning without `--seed` does not choose a new random seed. Small CPU/CUDA numerical differences can remain.

The displayed model's learned function, using the full stored coefficients without decimal rounding:

```text
h1 = ReLU(1.0502407550811768*x1 + 1.0502408742904663*x2 + 3.2880321668926626e-05)
h2 = ReLU(1.3059157133102417*x1 + 1.3059206008911133*x2 + -1.3059247732162476)

AND = -0.007523429114371538*h1 + 0.7761982083320618*h2 + 0.0046998863108456135
OR = 0.9376020431518555*h1 + -0.7469978928565979*h2 + 0.009954727254807949
NAND = -0.009855769574642181*h1 + -0.7538039684295654*h2 + 1.0069886445999146
XOR = 0.9427571892738342*h1 + -1.5201443433761597*h2 + 0.0068475413136184216
```

The terminal executable still plots raw full-table MSE, explicitly labelled log10 MSE, ReLU, and each trained gate along `x2=0` and `x2=1`. The two slices include all four Boolean corners. Intermediate inputs illustrate the continuous function implied by the learned parameters; no extra labels or interpolation accuracy are claimed there. Terminal slices use learned coefficients, not the exact reference. The README PNG instead displays the two-input decision boundaries. Plot code is `lib/easy_ai_learning/basic_nn/logic_report.rb`.

`model.json` saves full-precision parameters; `loss.json` saves every update, including the initial loss; `plots.txt` saves terminal plots. All files live under ignored `runs/learning/`. To reuse the saved parameters from the repository root:

```ruby
require_relative "learning/lib/easy_ai_learning/basic_nn/logic_network"
model = EasyAILearning::BasicNN::LogicNetwork.load(
  "runs/learning/basic_nn/logic-gates/model.json", device: :auto
)
p model.scores([1, 0]) # learned AND / OR / NAND / XOR scores
```

The load snippet assumes repository-root IRB. Keep the JSON output labels in their recorded order.

Tests compare every manually calculated gradient with finite differences and verify that random initialization learns all gates. These test the learning calculation, not just the final hand-written formulas.

Generate the PNG from a saved training run without retraining:

```bash
# Reproduce the current snapshot training (overwrites the default local run):
bundle exec ruby learning/01_basic_nn/train.rb --device auto --seed 20261002
bundle exec ruby learning/01_basic_nn/plot.rb
# Default PNG: runs/learning/basic_nn/logic-gates/figure/functions.png
# Export another seed for local comparison:
bundle exec ruby learning/01_basic_nn/plot.rb \
  --run runs/learning/basic_nn/logic-gates-3407 \
  --output runs/learning/basic_nn/logic-gates-3407/figure
# Publish a reviewed snapshot to this README:
cp runs/learning/basic_nn/logic-gates/figure/functions.png learning/01_basic_nn/images/logic-functions.png
```

The training executable uses Unicode plots without requiring gnuplot. PNG export is a separate optional step because Markdown needs a static image. Its code is `lib/easy_ai_learning/basic_nn/logic_figure.rb`. Autograd gradients are checked against both the scalar chain rule and finite differences.

## Artifact directories and reuse

```text
runs/learning/basic_nn/logic-gates/     ignored local run; current snapshot seed 20261002
|-- model.json                         architecture, device, hyperparameters, learned weights/biases
|-- loss.json                          full training history, including update zero
|-- plots.txt                          saved Unicode plots
`-- figure/                            optional gnuplot export and source data
    `-- functions.png

learning/01_basic_nn/images/logic-functions.png   versioned reviewed README snapshot
```

Keep only one representative README image, `images/logic-functions.png`, in version control. Per-seed figures and generated plot data stay under `runs/learning/`. The existing root `.gitignore` entry `/runs/` already ignores all training artifacts, including these small JSON weights. No extra ignore rule is needed. A different seed/run should use `--output runs/learning/basic_nn/logic-gates-2027` to preserve the previous files. The JSON stores inference parameters, not optimizer state for training resume. Later large models can use Torch binary tensors under the same ignored run hierarchy.

Use saved coefficients without running any training:

```bash
bundle exec ruby learning/01_basic_nn/predict.rb --device auto
bundle exec ruby learning/01_basic_nn/plot.rb
# For a different saved run:
bundle exec ruby learning/01_basic_nn/plot.rb \
  --run runs/learning/basic_nn/logic-gates-2027 \
  --output runs/learning/basic_nn/logic-gates-2027/figure
```

Both commands read the local saved model. Plot export deliberately loads it on CPU; the model's computation is the same, and figure metadata records the original training device. Training prefers CUDA; tests and tiny figure sampling can run on CPU without changing the model architecture.

## Verification

Torch.rb CUDA runs converge for seeds 1337 (1,623 updates), 2027 (1,293 updates) 3407 (1,801 updates) and 20261002 (1,271 updates), each with all 16 Boolean outputs correct and maximum absolute error below 0.01. Reloaded default weights give correct truth tables on CPU and CUDA without training. The PNG was regenerated from saved parameters and visually reviewed. Learning suite: 12 tests / 70 assertions, including finite-difference/manual-gradient/autograd agreement and moved GPT update/generation. Full repository RuboCop: 164 files clean. The moved GPT executable also completed a two-update CPU smoke run on a small local corpus; this validates execution, not text quality.
