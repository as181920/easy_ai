require "optparse"

module EasyAILearning
  module Course
    module Cli
      module_function

      def run(chapter, experiment)
        options = { seed: 1337, steps: 60, device: "auto", output: "runs/learning/#{chapter}/default" }
        OptionParser.new do |parser|
          parser.banner = "Usage: bundle exec ruby learning/#{chapter}/train.rb [options]"
          parser.on("--seed N", Integer) { |v| options[:seed] = v }
          parser.on("--steps N", Integer) { |v| options[:steps] = v }
          parser.on("--device NAME", "cpu, cuda, auto") { |v| options[:device] = v }
          parser.on("--output PATH") { |v| options[:output] = v }
        end.parse!
        context = Context.new(options)
        experiment.run(context)
        context.finish(chapter)
      end
    end
  end
end
