module DecisionScaling
  module_function

  def diagnostic_runtime(root, device: "cpu")
    verify(root)
    device = device.to_s
    raise ArgumentError, "Unknown diagnostic device" unless %w[cpu cuda].include?(device)
    paths = Dir[File.join(root, "arms/*/{screen,extended}.json")].sort
    rows = panel(root, "canonical").group_by { |row| row.fetch("language") }.values.map(&:first)
    values = paths.to_h do |path|
      result = JSON.parse(File.read(path))
      predictor = EasyAI::Decision::Predictor.load(result.fetch("checkpoint"), device: device, candidate_chunk_size: 1)
      raise "Diagnostic CUDA load fell back to CPU" unless predictor.device == device
      checks = rows.map do |row|
        options = row.fetch("options")
        first = predictor.probabilities(state: row.fetch("state"), question: row.fetch("question"), options: options)
        second = predictor.probabilities(state: row.fetch("state"), question: row.fetch("question"), options: options.reverse)
        probabilities = first.fetch("probabilities")
        json = JSON.parse(JSON.generate(first))
        delta = probabilities.keys.map { |key| (probabilities.fetch(key) - second.fetch("probabilities").fetch(key)).abs }.max
        { "language" => row.fetch("language"), "ids" => json.fetch("probabilities").keys.sort,
          "sum" => probabilities.values.sum, "finite" => probabilities.values.all?(&:finite?), "permutation_delta" => delta }
      end
      passed = checks.all? { |cell| cell.fetch("ids") == %w[no yes] && cell.fetch("finite") && (cell.fetch("sum") - 1.0).abs < 1e-6 && cell.fetch("permutation_delta") < 1e-4 }
      raise "Diagnostic serialization/permutation audit failed" unless passed
      name = File.basename(File.dirname(path)) + "/" + File.basename(path, ".json")
      puts "Diagnostic loading #{name}: #{predictor.device}, passed=#{passed}"
      $stdout.flush
      [name, { "passed" => passed, "device" => predictor.device, "weights_sha256" => sha(File.join(result.fetch("checkpoint"), "weights.pt")), "checks" => checks }]
    end
    write(File.join(root, "diagnostic-runtime-#{device}.json"), { "checkpoints" => values,
      "scope" => "Two bilingual binary observed-development requests per completed controlled endpoint. Checks checkpoint loading, finite normalized JSON probabilities and candidate permutation. No semantic acceptance, latency or throughput qualification." })
  end
end
