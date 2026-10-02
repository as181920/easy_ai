require "json"
require "fileutils"
require "unicode_plot"

module EasyAILearning
  module BasicNN
    class LogicReport
      def initialize(trainer:, seed:)
        @trainer, @model, @seed = trainer, trainer.model, seed
      end

      def write(directory)
        FileUtils.mkdir_p(directory)
        File.write(File.join(directory, "model.json"), JSON.pretty_generate(metadata) + "\n")
        File.write(File.join(directory, "loss.json"), JSON.pretty_generate(@trainer.history) + "\n")
        text = plots.map(&:to_s).join("\n\n")
        File.write(File.join(directory, "plots.txt"), text + "\n")
        text
      end

      def table
        header = "x1 x2 | target AND OR NAND XOR | trained scores | thresholded bits"
        lines = LogicGates::INPUTS.each_with_index.map do |input, row|
          scores = @model.scores(input)
          "#{input.join('  ')}   | #{LogicGates.targets[row].join('   ')} | " \
            "#{scores.map { |score| format('% .4f', score) }.join(' ')} | #{scores.map { |score| score >= 0.5 ? 1 : 0 }.join(' ')}"
        end
        ([header] + lines).join("\n")
      end

      def equations
        p = @model.parameter_values
        hidden = p[:hidden_weights].each_with_index.map do |weights, index|
          "h#{index + 1} = ReLU(#{format('%.6f', weights[0])}*x1 + #{format('%.6f', weights[1])}*x2 + #{format('%.6f', p[:hidden_biases][index])})"
        end
        outputs = p[:output_weights].each_with_index.map do |weights, index|
          "#{LogicGates::NAMES[index].upcase} = #{format('%.6f', weights[0])}*h1 + #{format('%.6f', weights[1])}*h2 + #{format('%.6f', p[:output_biases][index])}"
        end
        (hidden + outputs).join("\n")
      end

      private

      def metadata
        { architecture: [2, 2, 4], activation: "ReLU", outputs: LogicGates::NAMES,
          parameter_count: @model.parameter_count, seed: @seed, steps: @trainer.steps,
          learning_rate: @trainer.learning_rate, max_steps: @trainer.max_steps, tolerance: @trainer.tolerance,
          converged: @trainer.converged?, max_error: @trainer.max_error,
          initialization: "seeded random coefficients; positive hidden weights avoid initially dead units",
          scope: "Fits all four Boolean input combinations; raw linear outputs are not probabilities",
          device: @model.device, parameters: @model.parameter_values }
      end

      def plots
        steps, losses = @trainer.history.transpose
        raw = UnicodePlot.lineplot(steps, losses, title: "Full truth-table MSE (raw)", xlabel: "Update", ylabel: "MSE", width: 60, height: 10)
        logarithmic = UnicodePlot.lineplot(steps, losses.map { |loss| Math.log10([loss, 1e-16].max) },
          title: "Full truth-table MSE (log10; floor 1e-16)", xlabel: "Update", ylabel: "log10(MSE)", width: 60, height: 10)
        xs = (-20..20).map { |value| value / 10.0 }
        relu = UnicodePlot.lineplot(xs, xs.map { |value| @model.relu(value) }, title: "ReLU(x) = max(0, x)", width: 60, height: 10)
        [raw, logarithmic, relu] + LogicGates::NAMES.each_index.map { |gate| gate_plot(gate) }
      end

      def gate_plot(gate)
        xs = (0..40).map { |value| value / 40.0 }
        curves = [0, 1].map { |x2| xs.map { |x| @model.scores([x, x2])[gate] } }
        limits = [[0, *curves.flatten].min - 0.1, [1, *curves.flatten].max + 0.1]
        name = LogicGates::NAMES[gate].upcase
        plot = UnicodePlot.lineplot(xs, curves[0], name: "learned x2=0", color: :blue,
          title: "#{name}: learned function slices (only corners trained)", xlabel: "x1", ylabel: "score", width: 60, height: 10, ylim: limits)
        UnicodePlot.lineplot!(plot, xs, curves[1], name: "learned x2=1", color: :red)
        [0, 1].each do |x2|
          targets = [0, 1].map { |x1| LogicGates.call(name.downcase, x1, x2) }
          UnicodePlot.scatterplot!(plot, [0, 1], targets, name: "truth x2=#{x2}", color: x2.zero? ? :blue : :red)
        end
        plot
      end
    end
  end
end
