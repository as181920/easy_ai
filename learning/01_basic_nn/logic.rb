#!/usr/bin/env ruby
require_relative "../lib/easy_ai_learning/basic_nn/logic_gates"
require_relative "../lib/easy_ai_learning/basic_nn/scalar_logic_network"

gates = EasyAILearning::BasicNN::LogicGates
exact = EasyAILearning::BasicNN::ScalarLogicNetwork.exact
puts "Branches are the labels; perceptrons have fixed weights; exact ReLU weights prove existence."
puts "x1 x2 | if/else AND OR NAND XOR | perceptron AND OR NAND | exact ReLU AND OR NAND XOR"
gates::INPUTS.each do |left, right|
  branches = gates::NAMES.map { |name| gates.call(name, left, right) }
  perceptrons = %w[and or nand].map { |name| gates.perceptron(name, left, right) }
  puts "#{left}  #{right}  | #{branches.join('   ')} | #{perceptrons.join('   ')} | #{exact.forward([left, right]).join(' ')}"
end
puts "\nA single affine threshold cannot implement XOR."
puts "Exact network: h1=ReLU(x1+x2), h2=ReLU(x1+x2-1)."
puts "AND=h2; OR=h1-h2; NAND=1-h2; XOR=h1-2*h2."
