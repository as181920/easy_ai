module EasyAI
  module Decision
    module Data
      class Example
        attr_reader :id, :group_id, :language, :state, :question, :options, :target, :source, :contrast_group, :contrast_groups

        def initialize(row, require_target: true)
          row = row.transform_keys(&:to_s)
          @id = row.fetch("id", "request").to_s
          @group_id = row.fetch("group_id", @id).to_s
          @language = row.fetch("language", "und").to_s
          @source = row.fetch("source", "local").to_s
          @contrast_group = row["contrast_group"]
          validate_text!(@contrast_group) if @contrast_group
          @contrast_groups = row.fetch("contrast_groups", {})
          raise ArgumentError, "contrast_groups must be a mapping" unless @contrast_groups.is_a?(Hash)
          @contrast_groups = @contrast_groups.transform_keys(&:to_s)
          @contrast_groups.each do |key, value|
            validate_text!(key)
            validate_text!(value)
          end
          @state, @question = row.fetch("state"), row.fetch("question")
          @options = row.fetch("options").map { |option| option.transform_keys(&:to_s) }
          [@state, @question].each { |text| validate_text!(text) }
          raise ArgumentError, "At least two options required" if @options.length < 2
          @options.each do |option|
            option["id"] = normalize_option_id(option.fetch("id"))
            validate_text!(option.fetch("text"))
          end
          ids = @options.map { |option| option.fetch("id") }
          raise ArgumentError, "Option IDs must be unique" unless ids.uniq == ids
          @target = row["target"].nil? ? nil : normalize_option_id(row["target"])
          raise ArgumentError, "Target must identify a supplied option" if (require_target || @target) && !ids.include?(@target)
        end

        def target_index
          options.index { |option| option["id"] == target }
        end

        def texts
          [state, question] + options.map { |option| option["text"] }
        end

        def to_h
          result = { "id" => id, "group_id" => group_id, "language" => language, "source" => source,
           "state" => state, "question" => question, "options" => options, "target" => target }
          result["contrast_group"] = contrast_group if contrast_group
          result["contrast_groups"] = contrast_groups unless contrast_groups.empty?
          result
        end

        private

        def normalize_option_id(value)
          unless value.is_a?(String) || value.is_a?(Integer) || (value.is_a?(Float) && value.finite?)
            raise ArgumentError, "Option ID must be a string or finite number"
          end
          text = value.to_s
          validate_text!(text)
          text
        end

        def validate_text!(value)
          raise ArgumentError, "Expected nonempty UTF-8 string" unless value.is_a?(String) && !value.strip.empty? && value.valid_encoding?
        end
      end
    end
  end
end
