module DecisionBindingGeneralization
  module_function

  def publish_report(root)
    value = report(root)
    %w[namespace-correction/amendment namespace-correction/replay-audit exposure-confirmation/amendment selection-integrity/amendment reporting-integrity/amendment].each do |name|
      path = File.join(root, name + ".json")
      value[name] = JSON.parse(File.read(path)).except("runtime_source_sha256").merge("sha256" => sha(path)) if File.exist?(path)
    end
    value["source_integrity"] = "Frozen learning sources verified against the original protocol and explicitly chained runtime amendments. Numerical training changes are not implied by control-plane corrections."
    value["executed_trace_diagnostics"] = Dir[File.join(root, "arms/*/choice/training.jsonl")].sort.to_h do |path|
      rows = File.foreach(path).map { |line| JSON.parse(line) }
      next [path.delete_prefix(root + "/"), { "executed_updates" => 0 }] if rows.empty?
      checkpoint = EasyAI::Decision::Checkpoint.resolve(File.dirname(path))
      threshold = JSON.parse(File.read(File.join(checkpoint, "metadata.json"))).dig("config", "training", "grad_clip")
      norms = rows.map { |row| row.fetch("gradient_norm_preclip") }
      [path.delete_prefix(root + "/"), { "executed_updates" => rows.size, "mean_candidate_ce" => rows.sum { |row| row.fetch("choice_loss") }.fdiv(rows.size),
        "mean_pair_loss" => rows.sum { |row| row.fetch("binding_pair_loss") }.fdiv(rows.size), "mean_gradient_norm_preclip" => norms.sum.fdiv(norms.size),
        "gradient_clip_threshold" => threshold, "clipped_fraction" => norms.count { |norm| norm > threshold }.fdiv(norms.size),
        "scope" => "Executed updates, including replay. Objective weights and clipping can change optimization speed; matched steps are not matched convergence." }]
    end
    write(File.join(root, "report.json"), value)
    value
  end

  def plot(root)
    verify(root)
    directory = File.join(root, "plot")
    FileUtils.mkdir_p(directory)
    paths = Dir[File.join(root, "arms/*/development.json")].sort
    raise "No measured v07 histories" if paths.empty?
    series = paths.map.with_index do |path, index|
      trace = File.foreach(File.join(File.dirname(path), "choice/training.jsonl")).map { |line| JSON.parse(line) }.to_h { |row| [row.fetch("step"), row] }
      points = JSON.parse(File.read(path)).map do |row|
        step = row.fetch("step")
        window = trace.select { |number, _| number > step - 500 && number <= step }.values
        actor_min = ->(metric) { metric.fetch("by_language").values.map { |cell| cell.fetch("actor_binding_all_correct") }.min }
        [step, row.dig("training_probe", "worst_language_balanced"), row.dig("canonical", "worst_language_balanced"),
          actor_min.call(row.fetch("training_probe")), actor_min.call(row.fetch("canonical")),
          window.sum { |cell| cell.fetch("choice_loss") }.fdiv(window.size), window.sum { |cell| cell.fetch("gradient_norm_preclip") }.fdiv(window.size)].join(" ")
      end
      file = File.join(directory, "#{index}.dat")
      File.write(file, points.join("\n") + "\n")
      [File.basename(File.dirname(path)), file, index + 1]
    end
    curves = ->(column) { series.map { |name, file, color| "#{JSON.generate(file)} using 1:#{column} with linespoints lc #{color} title #{JSON.generate(name)}" }.join(", ") }
    output = File.expand_path("docs/images/decision-v07-generalization.png")
    script = <<~GP
      set terminal pngcairo size 1800,1600 font "Sans,11"
      set output #{JSON.generate(output)}
      set multiplot layout 3,2 margins 0.07,0.98,0.06,0.93 spacing 0.08,0.09 title "Binding generalization: observed development and training diagnostics"
      set grid
      set xlabel "Optimizer step (display deduplicates replay; ledger counts every update)"
      set yrange [0:1.05]
      set key bottom right
      set title "Worst-language fixed training-probe balanced accuracy"
      plot #{curves.call(2)}, 0.95 with lines dt 2 lc rgb "gray" title "Retention accuracy"
      set title "Worst-language canonical development balanced accuracy"
      plot #{curves.call(3)}
      set key top right
      set title "Worst-language fixed training-probe actor pairs (both correct)"
      plot #{curves.call(4)}, 0.85 with lines dt 2 lc rgb "gray" title "Development retention"
      set title "Worst-language canonical actor pairs (both correct)"
      plot #{curves.call(5)}, 0.25 with lines dt 2 lc rgb "gray" title "Independent-guess expectation"
      set autoscale y
      set title "Mean candidate CE over preceding 500 executed steps (replays deduplicated)"
      plot #{curves.call(6)}
      set title "Mean pre-clipping gradient norm over preceding 500 steps"
      plot #{curves.call(7)}
      unset multiplot
    GP
    path = File.join(directory, "figure.gp")
    File.write(path, script)
    raise "v07 gnuplot failed" unless system("gnuplot", path)
    puts "Wrote #{output}"
  end
end
