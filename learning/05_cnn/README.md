# 05 · CNN: convolutions, shared parameters, and feature maps

A minimal teaching experiment is implemented and runnable. Prerequisites: 01–03; review 04 for the convolutional AE. Next chapter: [06_resnet](../06_resnet/README.md).

Move from hand-calculated 2D cross-correlation to Torch Conv2d, then connect two convolutional layers into a classifier. Reuse the classification loss and optimizer from 02; the convolutional AE reuses the reconstruction objective from 04.

The data consists of independently generated 8×8 grayscale images: vertical or horizontal lines at random positions with light noise, with 64 training and 64 validation samples. Labels indicate line orientation. This simple graphics task does not represent MNIST/CIFAR or real-world visual generalization.

```text
[B,1,8,8] → Conv(1,4,3,pad=1) → ReLU → MaxPool(2)
            → Conv(4,8,3,pad=1) → ReLU → mean(H,W) → Linear(8,2)
```

The output spatial size is `floor((I+2P-K)/S)+1`; the parameter count is `C_out*C_in*K_h*K_w+C_out`. The classifier has 354 parameters, or 378 with BatchNorm; the flattened-input MLP has 806. The model uses global average pooling. It is a small teaching network inspired by LeNet, not a layer-by-layer reproduction of the original LeNet/AlexNet/VGG.

BatchNorm updates its running mean/variance using training batches and uses saved statistics in evaluation. These buffers are saved with the inference weights. CNN/MLP comparisons must also account for capacity and initialization differences; every model may score perfectly on this task.

Convolutional AE: `Conv(1,4)→ReLU→AvgPool(2)→nearest upsample→Conv(4,1)→sigmoid`. It reconstructs its input and outputs original/reconstructed images. Apply augmentation only during training and ensure that transformations preserve labels. A 90° rotation swaps the horizontal/vertical classes here, so retaining the original label would be incorrect.

Read the large-image, multilayer AlexNet/VGG architectures as historical extensions. Receptive fields, local connectivity, and parameter sharing are the core implementations in this chapter.

## Data, training, and independent inference

Run commands from the repository root and install dependencies with `bundle install`. Training and model inference default to `auto`: prefer CUDA and fall back to CPU when unavailable. You can also select `--device cpu` explicitly. These experiments use small synthetic datasets and require no model downloads. Limiting threads reduces CPU overhead for tiny tensors:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bundle exec ruby learning/05_cnn/data.rb
bundle exec ruby learning/05_cnn/train.rb --steps 60
bundle exec ruby learning/05_cnn/predict.rb
bundle exec rake test:learning
```

Use `--seed` to change the random seed and `--output` to separate experiment directories. Training also accepts `--device cpu/cuda/auto`. The default output is `runs/learning/05_cnn/default/`; rerunning overwrites artifacts with the same names. `data.rb` exports samples from the data recipe for inspection. The training entry point calls the generators directly and does not depend on that JSON file. Experiment-specific shifts and masks are documented in the experiment source and the actual `data.json`.

For inference, `--model PATH` selects a saved model. `--input PATH` accepts a JSON file containing `{"input": ...}`; the default is a small example with the required shape. Seq2seq/EncoderDecoder perform free-running generation, GPT uses top-1 generation, and other models return scores or reconstructions. RL actor-critic models return policy logits and a value estimate.

## Results and correctness

The [recorded run](results.json) includes the seed, step count, environment, and metrics. The figure below comes from that run; it is not a test acceptance threshold.

![Chapter experiment results](images/image-reconstruction.svg)

[Experiment code](../lib/easy_ai_learning/cnn/experiment.rb) connects the steps; shared data generators are in [course/data.rb](../lib/easy_ai_learning/course/data.rb). Core checks are in the [tests](../test/course/vision_test.rb), with additional gradient comparisons in [derivatives_test.rb](../test/course/derivatives_test.rb). Tests check deterministic formulas, shapes, masks, gradients, state, and parameter updates. They do not train toward a required accuracy or weight distribution.

Artifacts include the actual data, JSON inference state, history, diagnostics, and SVG figures. Full local parameters and histories remain under the ignored `runs/` directory; the repository contains only compact result summaries and figures. The non-neural experiments in 00/07 and the interactive training in 15 use task-specific records rather than forcing every result into a classification-loss format.
