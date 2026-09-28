require "open3"

module EasyAI
  module Runtime
    class DevicePolicy
      class MemoryBudgetExceeded < StandardError; end

      attr_reader :requested, :budget_mib

      def initialize(requested: "auto", budget_mib: 4096, logger: EasyAI.logger)
        raise ArgumentError, "Unknown device #{requested}" unless %w[auto cpu cuda].include?(requested)
        @requested, @budget_mib, @logger = requested, budget_mib, logger
      end

      def resolve
        return "cpu" if requested == "cpu"
        unless Torch::CUDA.available?
          @logger.warn("CUDA unavailable; using CPU")
          return "cpu"
        end
        Torch.zeros([1], device: "cuda").sum.item
        "cuda"
      rescue Torch::Error => error
        raise unless recoverable?(error)
        @logger.warn("CUDA initialization failed: #{error.message.lines.first}; using CPU")
        "cpu"
      end

      def recoverable?(error)
        error.is_a?(MemoryBudgetExceeded) || (error.is_a?(Torch::Error) &&
          error.message.match?(/CUDA.*out of memory|CUDA.*initialization error|no NVIDIA driver|insufficient driver|no CUDA GPUs are available/i))
      end

      def check_budget!(device)
        return unless device.to_s.start_with?("cuda")
        used = process_memory_mib
        raise MemoryBudgetExceeded, "GPU process uses #{used} MiB; budget #{budget_mib} MiB" if used && used > budget_mib
        used
      end

      def process_memory_mib
        output, status = Open3.capture2("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits", err: File::NULL)
        return unless status.success?
        values = output.lines.filter_map do |line|
          pid, memory = line.strip.split(/,\s*/)
          Integer(memory, exception: false) if pid.to_i == Process.pid
        end
        values.max
      rescue Errno::ENOENT
        nil
      end
    end
  end
end
