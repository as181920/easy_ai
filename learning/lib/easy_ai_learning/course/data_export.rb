module EasyAILearning
  module Course
    module DataExport
      module_function

      def run(chapter, options)
        seed = options.fetch(:seed)
        splits = { train: seed, validation: seed + 1, test: seed + 2 }.transform_values do |split_seed|
          case chapter
          when "00_foundations" then { regression: Data.regression(seed: split_seed), classification: Data.classification(seed: split_seed), manifold: Data.manifold(seed: split_seed) }
          when "01_basic_nn" then { inputs: BasicNN::LogicGates::INPUTS, targets: BasicNN::LogicGates.targets, note: "Complete truth table; identical reference in each split, not generalization data." }
          when "02_training", "03_diagnostics", "14_transfer_learning" then Data.classification(seed: split_seed)
          when "04_autoencoder" then Data.manifold(seed: split_seed)
          when "05_cnn", "06_resnet", "16_capstone" then Data.images(seed: split_seed)
          when "08_rnn", "12_gpt" then Data.sequences(seed: split_seed, length: chapter == "12_gpt" ? 8 : 5)
          when "09_seq2seq", "10_attention", "11_transformer" then Data.reversal(seed: split_seed)
          when "13_generative" then { manifold: Data.manifold(seed: split_seed), mixture: Data.mixture(seed: split_seed), images: Data.images(seed: split_seed), tokens: Data.sequences(seed: split_seed) }
          when "07_tokenizers" then { text: split_seed == seed ? "hello world\n你好 世界\nhello learning\n你好 learning\n" * 8 : "hello 世界\nnew words! 你好\t🙂" }
          when "15_rl" then { environment: "Five-state chain; left/right; goal reward 1, other transitions -0.02; time limit 12.", rollout_seed: split_seed }
          else raise ArgumentError, "Unknown chapter"
          end
        end
        artifact = Artifacts.new(options.fetch(:output))
        artifact.json("generated-data", { chapter: chapter, seed: seed, splits: splits, source: "Local synthetic generators, no downloads." })
      end
    end
  end
end
