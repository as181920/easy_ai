require "json"
require "fileutils"
require "cgi"

module EasyAILearning
  module Course
    class Artifacts
      attr_reader :directory

      def initialize(directory)
        @directory = directory
        FileUtils.mkdir_p(directory)
      end

      def json(name, value)
        File.write(File.join(directory, "#{name}.json"), JSON.pretty_generate(value) + "\n")
      end

      # Vector output needs no plotting package; values and axis ranges stay visible.
      def plot(name, series, title: name)
        points = series.values.flatten(1)
        return if points.empty?

        xs, ys = points.transpose
        xmin, xmax = xs.minmax
        ymin, ymax = ys.minmax
        xmax = xmin + 1 if xmax == xmin
        ymax = ymin + 1 if ymax == ymin
        colors = %w[#2563eb #dc2626 #059669 #9333ea #d97706 #0891b2]
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="420" viewBox="0 0 760 420">',
          '<rect width="760" height="420" fill="white"/>',
          %(<text x="55" y="25" font-size="16">#{CGI.escapeHTML(title)}</text>),
          '<path d="M55 45V340H720" fill="none" stroke="black"/>',
          %(<text x="55" y="360" font-size="12">x: #{xmin.round(4)} … #{xmax.round(4)}; y: #{ymin.round(6)} … #{ymax.round(6)}</text>)]
        series.each_with_index do |(label, values), index|
          color = colors[index % colors.size]
          xy = values.map { |x, y| "#{55 + 665 * (x - xmin) / (xmax - xmin).to_f},#{340 - 285 * (y - ymin) / (ymax - ymin).to_f}" }
          svg << %(<polyline points="#{xy.join(' ')}" fill="none" stroke="#{color}" stroke-width="2"/>)
          svg << %(<text x="#{55 + (index % 4) * 170}" y="#{385 + (index / 4) * 15}" fill="#{color}" font-size="11">#{CGI.escapeHTML(label.to_s)}</text>)
        end
        File.write(File.join(directory, "#{name}.svg"), (svg + ["</svg>"]).join("\n"))
      end

      def scatter(name, groups, title: name)
        all = groups.values.flatten(1)
        xmin, xmax = all.map(&:first).minmax
        ymin, ymax = all.map { |p| p[1] }.minmax
        xmax = xmin + 1 if xmax == xmin
        ymax = ymin + 1 if ymax == ymin
        colors = %w[#2563eb #dc2626 #059669 #9333ea]
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400">', '<rect width="640" height="400" fill="white"/>',
          %(<text x="30" y="25" font-size="16">#{CGI.escapeHTML(title)}</text>),
          %(<text x="30" y="380" font-size="12">x: #{xmin.round(3)} … #{xmax.round(3)}; y: #{ymin.round(3)} … #{ymax.round(3)}</text>)]
        groups.each_with_index do |(label, points), index|
          color = colors[index % colors.size]
          points.each do |x, y, *_|
            cx, cy = 40 + (x - xmin) / (xmax - xmin) * 550, 335 - (y - ymin) / (ymax - ymin) * 285
            svg << %(<circle cx="#{cx}" cy="#{cy}" r="3" fill="#{color}" opacity="0.65"/>)
          end
          svg << %(<text x="#{30 + index * 145}" y="355" font-size="12" fill="#{color}">#{CGI.escapeHTML(label.to_s)}</text>)
        end
        File.write(File.join(directory, "#{name}.svg"), (svg + ["</svg>"]).join("\n"))
      end

      def image_grid(name, groups, count: 3)
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="380" height="#{groups.size * 85 + 30}">',
          '<rect width="100%" height="100%" fill="white"/>']
        groups.each_with_index do |(label, images), group|
          svg << %(<text x="5" y="#{group * 85 + 20}" font-size="12">#{CGI.escapeHTML(label.to_s)}</text>)
          images.first(count).each_with_index do |image, column|
            image = image.first if image.size == 1
            image.each_with_index do |row, r|
              row.each_with_index do |value, c|
                intensity = ([[value, 0.0].max, 1.0].min * 255).round
                svg << %(<rect x="#{140 + column * 75 + c * 8}" y="#{group * 85 + 5 + r * 8}" width="8" height="8" fill="rgb(#{intensity},#{intensity},#{intensity})"/>)
              end
            end
          end
        end
        File.write(File.join(directory, "#{name}.svg"), (svg + ["</svg>"]).join("\n"))
      end

      def save_model(name, model, config: {})
        json(name, { format: 1, class: model.class.name, config: config,
          state: model.state_dict.transform_values { |t| { shape: t.shape, values: t.detach.cpu.to_a, dtype: t.dtype.to_s } } })
      end

      def self.load_model(path, model)
        saved = JSON.parse(File.read(path))
        raise ArgumentError, "Incompatible model artifact" unless saved.fetch("format") == 1 && saved.fetch("class") == model.class.name

        expected = model.state_dict
        raise ArgumentError, "State keys differ" unless expected.keys.sort == saved.fetch("state").keys.sort

        tensors = expected.to_h do |name, tensor|
          entry = saved.fetch("state").fetch(name)
          raise ArgumentError, "Wrong shape for #{name}" unless entry.fetch("shape") == tensor.shape

          [name, Torch.tensor(entry.fetch("values"), dtype: tensor.dtype, device: tensor.device)]
        end
        model.load_state_dict(tensors)
        model.eval
      end
    end
  end
end
