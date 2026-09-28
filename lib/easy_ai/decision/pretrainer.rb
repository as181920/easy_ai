module EasyAI
  module Decision
    class Pretrainer < Trainer
      def initialize(**options)
        super(**options, task: :mlm)
      end
    end
  end
end
