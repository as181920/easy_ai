module EasyAILearning
  module BasicNN
    # Branches define the truth table; they are never called by neural inference.
    module LogicGates
      NAMES = %w[and or nand xor].freeze
      INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]].freeze
      PERCEPTRONS = {
        "and" => { weights: [1, 1], bias: -1.5 },
        "or" => { weights: [1, 1], bias: -0.5 },
        "nand" => { weights: [-1, -1], bias: 1.5 }
      }.freeze

      module_function

      def call(name, left, right)
        validate_bits!(left, right)
        case name.to_s
        when "and" then and_gate(left, right)
        when "or" then or_gate(left, right)
        when "nand" then nand_gate(left, right)
        when "xor" then xor_gate(left, right)
        else raise ArgumentError, "Unknown gate: #{name}"
        end
      end

      def and_gate(left, right)
        if left == 1 && right == 1
          1
        else
          0
        end
      end

      def or_gate(left, right)
        if left == 1 || right == 1
          1
        else
          0
        end
      end

      def nand_gate(left, right)
        if left == 1 && right == 1
          0
        else
          1
        end
      end

      def xor_gate(left, right)
        if left != right
          1
        else
          0
        end
      end

      def perceptron(name, left, right)
        validate_bits!(left, right)
        config = PERCEPTRONS.fetch(name.to_s) { raise ArgumentError, "XOR is not a single perceptron gate" }
        score = config[:weights][0] * left + config[:weights][1] * right + config[:bias]
        score >= 0 ? 1 : 0
      end

      def targets
        INPUTS.map { |left, right| [call("xor", left, right)] }
      end

      def validate_bits!(left, right)
        raise ArgumentError, "Inputs must be 0 or 1" unless [left, right].all? { |bit| bit == 0 || bit == 1 }
      end
    end
  end
end
