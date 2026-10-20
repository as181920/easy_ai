module DecisionScaling
  module_function

  def parameter_audit(root)
    verify(root)
    paths = Dir[File.join(root, "arms/*/{screen,extended}.json")].sort
    raise "No completed controlled endpoints" if paths.empty?
    values = paths.to_h do |path|
      cell = File.basename(File.dirname(path))
      profile, _, seed = cell.split("-")
      result = JSON.parse(File.read(path))
      loaded = EasyAI::Decision::Checkpoint.load(result.fetch("checkpoint"))
      original, = initialize_model(profile, microbatch: 16, seed: Integer(seed))
      baseline = original.named_parameters
      tensors = Torch.no_grad do
        loaded.fetch(:model).named_parameters.to_h do |name, tensor|
          before = baseline.fetch(name)
          delta = tensor - before
          [name, { "elements" => tensor.numel, "initial_rms" => before.square.mean.sqrt.item, "final_rms" => tensor.square.mean.sqrt.item,
            "delta_rms" => delta.square.mean.sqrt.item, "maximum_absolute_change" => delta.abs.max.item,
            "nonzero_change_fraction" => delta.ne(0).sum.item.fdiv(tensor.numel), "initially_zero" => before.abs.max.item.zero? }]
        end
      end
      raise "Nonfinite parameter audit" unless tensors.values.all? { |row| row.values.grep(Numeric).all?(&:finite?) }
      key = "#{cell}/#{File.basename(path, '.json')}"
      puts "Parameter audit #{key}: #{tensors.size} tensors"
      $stdout.flush
      [key, { "checkpoint_weights_sha256" => loaded.fetch(:weights_fingerprint), "result_sha256" => sha(path), "parameters" => result.fetch("parameters"), "tensors" => tensors }]
    end
    write(File.join(root, "parameter-audit.json"), { "endpoints" => values,
      "scope" => "CPU tensor comparisons with the same seeded function-preserving initialization. No optimizer update or prediction. Nonzero changes establish parameter movement, not causal usefulness, gradient magnitude or semantic generalization; weight distributions alone cannot certify correctness." })
  end
end
