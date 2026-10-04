require "jekyll-watch"

module CourseSite
  # Jekyll normally watches only its source directory. Lessons stay outside it.
  module Watcher
    private
      def build_listener(site, options)
        Listen.to(
          File.join(ROOT, "learning"), File.join(ROOT, "docs"),
          ignore: [%r{\Acourse_site/(?:\.jekyll-cache|\.sass-cache|\.jekyll-metadata)(?:/|\z)}],
          force_polling: options["force_polling"],
          &listen_handler(site)
        )
      end
  end
end

Jekyll::Watcher.singleton_class.prepend(CourseSite::Watcher)
