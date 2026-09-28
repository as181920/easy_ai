require "rake/testtask"

Rake::TestTask.new(:test) do |task|
  task.libs << "test"
  task.pattern = "test/**/*_test.rb"
end

Rake::TestTask.new("test:learning") do |task|
  task.libs << "lib"
  task.libs << "learning/lib"
  task.pattern = "learning/test/**/*_test.rb"
end

desc "Run Ruby style checks"
task :lint do
  sh "bundle exec rubocop --cache-root tmp/rubocop"
end

task default: :test
