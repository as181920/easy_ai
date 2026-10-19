module DecisionRelational
  module_function

  def generalization_plot(root)
    directory = File.join(root, "plot-generalization")
    FileUtils.mkdir_p(directory)
    series = Dir.glob(File.join(root, "pilot*/seed-*/development.json")).sort.each_with_index.map do |path, index|
      history = JSON.parse(File.read(path))
      label = File.basename(File.dirname(File.dirname(path))).sub("pilot", "standard").sub("standard-focused", "focused") + "/" + File.basename(File.dirname(path)).delete_prefix("seed-")
      data = history.map do |cell|
        training = cell.fetch("training_probe").fetch("by_language").values.map { |value| value.fetch("balanced") }.min
        binary = cell.fetch("profiles").fetch("binary")
        controlled = binary.fetch("controlled")
        groups = controlled.fetch("by_language").values.flat_map { |value| %w[actor event].map { |kind| value.fetch("#{kind}_binding_all_correct") } }.min
        natural = binary.fetch("natural").fetch("by_language").values.map { |value| value.fetch("accuracy") }.min
        joint = cell.fetch("profiles")["joint"]
        [cell.fetch("step"), training, controlled.fetch("worst_language_balanced"), groups, natural,
          joint ? joint.dig("controlled", "worst_language_balanced") : "NaN"].join(" ")
      end
      file = File.join(directory, "series-#{index}.dat")
      File.write(file, data.join("\n") + "\n")
      { label: label, path: file, color: index + 1 }
    end
    raise "No supervised development measurements" if series.empty?
    quote = ->(value) { '"' + value.to_s.gsub('\\', '\\\\').gsub('"', '\\"') + '"' }
    curves = lambda do |column|
      series.map { |item| "#{quote.call(item.fetch(:path))} using 1:#{column} with linespoints lc #{item.fetch(:color)} title #{quote.call(item.fetch(:label))}" }.join(", \\\n")
    end
    output = File.expand_path("docs/images/decision-v05-generalization.png")
    script = <<~GP
      set terminal pngcairo size 1800,1200 font "Sans,11"
      set output #{quote.call(output)}
      set multiplot layout 2,2 margins 0.07,0.98,0.08,0.92 spacing 0.08,0.13 title "Decision supervised diagnostics: training probes and observed development (not fresh acceptance)"
      set grid
      set xlabel "Optimizer updates"
      set yrange [0:1.05]
      set key bottom right
      set title "Worst-language balanced training-probe accuracy"
      set ylabel "Balanced accuracy"
      plot #{curves.call(2)}, 0.95 with lines dt 2 lc rgb "gray" title "Known mastery accuracy floor"
      set title "Worst-language canonical binary development accuracy"
      plot #{curves.call(3)}, 0.75 with lines dt 2 lc rgb "gray" title "Binary eligibility accuracy floor"
      set title "Worst actor/event complete-group correctness (binary development)"
      set ylabel "All-correct fraction"
      plot #{curves.call(4)}, 0.5 with lines dt 2 lc rgb "gray" title "Binary eligibility group floor"
      set title "Worst-language natural binary development accuracy"
      set ylabel "Accuracy"
      plot #{curves.call(5)}, 0.65 with lines dt 2 lc rgb "gray" title "Natural eligibility floor"
      unset multiplot
    GP
    File.write(File.join(directory, "figure.gp"), script)
    IO.popen(["gnuplot"], "w") { |process| process.write(script) }
    raise "gnuplot failed" unless $?.success?
    puts "Wrote #{output}"
  end
end
