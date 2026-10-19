module DecisionRelational
  module_function

  def plot(root)
    ledger = report(root)
    directory = File.join(root, "plot")
    FileUtils.mkdir_p(directory)
    series = ledger.fetch("fits").map.with_index do |(name, result), index|
      path = File.join(directory, "#{name}.dat")
      points = result.fetch("measurements").map do |cell|
        languages = cell.fetch("by_language").values
        [cell.fetch("step"), languages.map { |value| value.fetch("balanced") }.min,
          languages.map { |value| value.fetch("actor_binding_all_correct") }.min, cell.fetch("training_loss")].join(" ")
      end
      File.write(path, points.join("\n") + "\n")
      [name, path, index + 1]
    end
    dev = File.join(directory, "development.dat")
    File.write(dev, ledger.fetch("fits").map do |name, result|
      label = name.delete_suffix("-1337").gsub("separate", "sep").gsub("position-scale", "scale").gsub("candidate", "cand").gsub("rotary", "RoPE")
      [label, *%w[en-US zh-CN].map { |language| result.dig("observed_development", "by_language", language, "balanced") }].join(" ")
    end.join("\n") + "\n")
    figure = File.expand_path("docs/images/decision-v05-relational.png")
    quote = ->(value) { '"' + value.to_s.gsub('\\', '\\\\').gsub('"', '\\"') + '"' }
    curves = ->(column) { series.map { |name, path, color| "#{quote.call(path)} using 1:#{column} with linespoints lc #{color} title #{quote.call(name)}" }.join(", \\\n") }
    script = <<~GP
      set terminal pngcairo size 1800,1200 font "Sans,11"
      set output #{quote.call(figure)}
      set multiplot layout 2,2 margins 0.07,0.98,0.16,0.92 spacing 0.08,0.13 title "Decision relational diagnostics: training fit and observed development (not fresh acceptance)"
      set key font ",9"
      set grid
      set xlabel "Optimizer updates"
      set yrange [0:1.05]
      set title "Worst-language balanced training accuracy"
      set ylabel "Balanced accuracy"
      plot #{curves.call(2)}
      set title "Worst-language mixed-actor complete-group correctness (training)"
      set ylabel "All-correct fraction"
      plot #{curves.call(3)}
      set title "Sampled training CE at checks (unsmoothed)"
      set ylabel "Cross-entropy"
      set yrange [0:*]
      plot #{curves.call(4)}
      set title "Final observed-development balanced accuracy"
      set xlabel "Recipe"
      set ylabel "Balanced accuracy"
      set yrange [0:1.05]
      set style data histograms
      set style histogram clustered gap 1
      set style fill solid 0.8 border -1
      set xtics rotate by -25 font ",10"
      plot #{quote.call(dev)} using 2:xtic(1) title "English", '' using 3 title "Chinese"
      unset multiplot
    GP
    File.write(File.join(directory, "figure.gp"), script)
    IO.popen(["gnuplot"], "w") { |process| process.write(script) }
    raise "gnuplot failed" unless $?.success?
    puts "Wrote #{figure}"
  end
end
