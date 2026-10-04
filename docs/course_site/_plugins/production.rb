require "uri"

Jekyll::Hooks.register :site, :after_init do |site|
  url = ENV["LEARNING_SITE_URL"].to_s
  if Jekyll.env == "production" && url.empty?
    raise Jekyll::Errors::FatalException, "Set LEARNING_SITE_URL to the public site origin for a production build"
  end
  unless url.empty?
    origin = URI.parse(url)
    unless %w[http https].include?(origin.scheme) && origin.host && ["", "/"].include?(origin.path) &&
        !origin.userinfo && !origin.query && !origin.fragment
      raise Jekyll::Errors::FatalException, "LEARNING_SITE_URL must be an HTTP(S) origin without a path, query, or credentials"
    end
    site.config["url"] = url.delete_suffix("/")
  end
  if ENV.key?("LEARNING_SITE_BASEURL")
    prefix = ENV.fetch("LEARNING_SITE_BASEURL").delete_suffix("/")
    unless prefix.empty? || (prefix.match?(%r{\A/[a-zA-Z0-9_/-]+\z}) && !prefix.include?("..") && !prefix.include?("//"))
      raise Jekyll::Errors::FatalException, "LEARNING_SITE_BASEURL must be empty or a path such as /course"
    end
    site.config["baseurl"] = prefix
    site.baseurl = prefix
  end
end
