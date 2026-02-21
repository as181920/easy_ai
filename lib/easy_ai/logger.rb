require "fileutils"
require "active_support/logger"
require "active_support/core_ext/string/filters"

module EasyAI
  module Logger
    module_function

    def logger
      @logger ||= build_logger
    end

    def build_logger
      path = ENV["EASY_AI_LOG_PATH"] || ENV["LOG_PATH"] || default_log_path
      level_name = (ENV["EASY_AI_LOG_LEVEL"] || ENV["LOG_LEVEL"] || "INFO").upcase

      io = path.to_s.casecmp("stdout").zero? ? STDOUT : ensure_log_io(path)
      logger = ActiveSupport::Logger.new(io)
      logger.level = resolve_level(level_name)
      logger.progname = "EasyAI"
      logger.formatter = proc do |severity, datetime, progname, msg|
        timestamp = datetime.getlocal.strftime("%Y-%m-%d %H:%M:%S.%L %Z")
        "#{timestamp} [#{severity}] #{progname}: #{msg}\n"
      end
      logger
    end

    def reset!
      @logger&.close if closable_logger?(@logger)
      @logger = nil
    end

    def summary(value, limit: 180, literal: false)
      content = literal ? value.inspect : normalized_text(value.to_s)
      content.truncate(limit)
    end

    def default_log_path
      File.expand_path("../../log/train.log", __dir__)
    end

    def ensure_log_io(path)
      FileUtils.mkdir_p(File.dirname(path))
      path
    end

    def resolve_level(name)
      ActiveSupport::Logger.const_get(name)
    rescue NameError
      ActiveSupport::Logger::INFO
    end

    def closable_logger?(logger)
      logger && logger.respond_to?(:close) && logger.instance_variable_get(:@logdev)&.dev&.respond_to?(:close)
    end

    def normalized_text(text)
      text.gsub(/\r?\n+/, " ").squeeze(" ").strip
    end
  end

  def self.logger
    Logger.logger
  end
end
