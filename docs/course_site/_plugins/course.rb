require "cgi"
require "kramdown"
require "kramdown-parser-gfm"
require "rouge"
require "uri"

module CourseSite
  ROOT = File.expand_path("../../..", __dir__)
  EXTRA = %w[docs/learning-course-print.md docs/learning-site.md].freeze

  class Generator < Jekyll::Generator
    safe true
    priority :low

    def generate(site)
      @site = site
      @documents = ["learning/README.md", *Dir.glob("learning/[0-9][0-9]_*/README.md", base: ROOT).sort, *EXTRA]
      @assets = Dir.glob("learning/**/*", base: ROOT).select do |path|
        File.file?(File.join(ROOT, path)) && %w[.rb .json .svg .png .jpg .jpeg .webp].include?(File.extname(path))
      end
      @documents.each_with_index { |path, index| document_page(path, index) }
      @assets.each { |path| asset(path) }
    end

    private
      def document_page(path, index)
        source = File.read(File.join(ROOT, path))
        title = source.lines.first.sub(/\A#\s+/, "").strip
        source = source.sub(/\A([^\n]+)\n/, "\\1\n{: .no_toc }\n\n## On this page\n{: .no_toc }\n\n1. TOC\n{:toc}\n")
        document = Kramdown::Document.new(source, input: "GFM", syntax_highlighter: "rouge")
        rewrite_links(document.root, path)
        page = add_page(path, title, document.to_html, "nav_order" => index)
        previous, following = @documents[index - 1], @documents[index + 1]
        page.data["previous_course"] = destination(previous) if index.positive?
        page.data["next_course"] = destination(following) if following
      end

      def rewrite_links(element, source)
        attribute = element.type == :a ? "href" : element.type == :img ? "src" : nil
        element.attr[attribute] = resolve_link(element.attr[attribute], source) if attribute
        element.children.each { |child| rewrite_links(child, source) }
      end

      def resolve_link(link, source)
        uri = URI.parse(link.gsub(/[^\x21-\x7e]/) { |character| URI::DEFAULT_PARSER.escape(character) })
        return link if uri.scheme || uri.host || uri.path.to_s.empty?

        path = File.expand_path(URI::DEFAULT_PARSER.unescape(uri.path), File.dirname(File.join(ROOT, source)))
        repository_path = path.delete_prefix(ROOT + "/")
        unless path.start_with?(ROOT + "/") && (@documents.include?(repository_path) || @assets.include?(repository_path))
          raise Jekyll::Errors::FatalException, "Unresolved course link in #{source}: #{link}"
        end
        url = @documents.include?(repository_path) || repository_path.end_with?(".rb") ? destination(repository_path) : "/#{repository_path}"
        url = @site.baseurl.to_s + url
        url += "?#{uri.query}" if uri.query
        url += "##{uri.fragment}" if uri.fragment
        url
      end

      def destination(path)
        return "/index.html" if path == "learning/README.md"
        return "/#{path}.html" if path.end_with?(".rb")
        "/#{path.sub(/\.md\z/, '.html')}"
      end

      def add_page(path, title, content, data = {})
        url = destination(path)
        page = Jekyll::PageWithoutAFile.new(@site, @site.source, File.dirname(url).delete_prefix("/"), File.basename(url))
        page.content = content
        page.data.merge!({ "layout" => "default", "title" => title, "permalink" => url, "render_with_liquid" => false }.merge(data))
        @site.pages << page
        page
      end

      def asset(path)
        @site.static_files << Jekyll::StaticFile.new(@site, ROOT, File.dirname(path), File.basename(path))
        return unless path.end_with?(".rb")

        source = File.read(File.join(ROOT, path))
        highlighted = Rouge.highlight(source, "ruby", "html")
        content = "<h1>#{CGI.escapeHTML(File.basename(path))}</h1><p>Source: <code>#{CGI.escapeHTML(path)}</code>. " \
          "<a href='#{@site.baseurl}/#{path}'>Download original</a></p><div class='highlight'><pre>#{highlighted}</pre></div>"
        add_page(path, File.basename(path), content, "nav_exclude" => true, "search_exclude" => true)
      end
  end
end
