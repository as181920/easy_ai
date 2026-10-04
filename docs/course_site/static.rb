require "optparse"
require "webrick"

root = File.expand_path("../..", __dir__)
port = 8001
OptionParser.new { |parser| parser.on("--port PORT", Integer) { |value| port = value } }.parse!
directory = File.join(root, "tmp/learning-site")
abort "Build the site first: bin/learning-docs build" unless File.file?(File.join(directory, "index.html"))
server = WEBrick::HTTPServer.new(Port: port, BindAddress: "127.0.0.1", DocumentRoot: directory)
%w[INT TERM].each { |signal| Signal.trap(signal) { server.shutdown } }
server.start
