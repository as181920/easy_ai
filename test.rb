require_relative "bpe_tokenizer"

# Get text from specified dir
#
# Parse and Prepare text for tokenize

corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest widest"
tokenizer = BpeTokenizer.new
tokenizer.train(corpus)

puts "\nTokenizing 'lower':"
p tokenizer.tokenize("lower")

# Train tokenizer
#
# Continue
