require_relative "../test_helper"
require "tmpdir"

class SequenceTest < Minitest::Test
  def test_reversible_bpe_handles_unseen_utf8_and_whitespace_after_save_load
    model = EasyAILearning::Tokenizers::ReversibleBpe.new.train("aaaa 你好\n" * 8)
    text = " unseen\t🙂\n你好  "

    assert_equal text, model.decode(model.encode(text))
    Dir.mktmpdir do |directory|
      path = File.join(directory, "tokenizer.json")
      model.save(path)
      restored = EasyAILearning::Tokenizers::ReversibleBpe.load(path)

      assert_equal model.encode(text), restored.encode(text)
      assert_equal text, restored.decode(restored.encode(text))
    end
  end

  def test_rnn_lstm_gru_follow_hand_written_gate_equations
    x, h = Torch.tensor([[1.0]]), Torch.tensor([[0.4]])
    %i[rnn lstm gru].each do |kind|
      cell = EasyAILearning::RNN::Cell.new(input: 1, hidden: 1, kind: kind)
      Torch.no_grad { cell.parameters.each { |p| p.fill!(0.2) } }
      input, recurrent = 0.4, 0.28
      sigmoid = EasyAILearning::Foundations::Math.method(:sigmoid)
      expected = case kind
                 when :rnn then Math.tanh(input + recurrent)
                 when :lstm
                   gate = sigmoid.call(input + recurrent)
                   memory = gate * 0.3 + gate * Math.tanh(input + recurrent)
                   gate * Math.tanh(memory)
                 when :gru
                   gate = sigmoid.call(input + recurrent)
                   (1 - gate) * Math.tanh(input + gate * recurrent) + gate * 0.4
                 end
      actual = cell.call(x, kind == :lstm ? [h, Torch.tensor([[0.3]])] : h)
      actual = actual.first if actual.is_a?(Array)

      assert_in_delta expected, actual.item, 1e-6
    end
  end

  def test_recurrent_weight_derivative_matches_finite_difference
    cell = EasyAILearning::RNN::Cell.new(input: 1, hidden: 1)
    Torch.no_grad { cell.parameters.each { |p| p.fill!(0.2) } }
    x, h = Torch.tensor([[0.3]]), Torch.tensor([[0.5]])
    cell.call(x, h).sum.backward
    analytic = cell.input_projection.weight.grad.item
    original = cell.input_projection.weight.item
    numerical = EasyAILearning::Foundations::Math.finite_difference(original, epsilon: 0.001) do |value|
      Torch.no_grad { cell.input_projection.weight.fill!(value) }
      cell.call(x, h).item
    end

    assert_in_delta analytic, numerical, 5e-5
  end

  def test_padding_preserves_recurrent_state
    model = EasyAILearning::RNN::Model.new
    a, b = Torch.tensor([[1, 2, 3]], dtype: :int64), Torch.tensor([[1, 4, 5]], dtype: :int64)
    lengths = Torch.tensor([1], dtype: :int64)

    assert_equal model.call(a, lengths: lengths).to_a, model.call(b, lengths: lengths).to_a
  end

  def test_teacher_input_is_right_shifted_and_decoder_has_no_future_dependency
    source, decoder, target = EasyAILearning::Course::Data.reversal(count: 2)

    assert_equal [1] + target.first[0...-1], decoder.first
    model = EasyAILearning::Seq2seq::Model.new
    src = Torch.tensor(source, dtype: :int64)
    tokens = Torch.tensor(decoder, dtype: :int64)
    expected = model.call(src, tokens)
    changed = tokens.clone
    changed[0][4] = 3
    actual = model.call(src, changed)

    assert_equal expected.narrow(1, 0, 4).to_a, actual.narrow(1, 0, 4).to_a
  end
end

class BpttTest < Minitest::Test
  def test_truncation_stops_gradient_across_time_boundary
    model = EasyAILearning::RNN::Model.new(vocab: 3, hidden: 2)
    Torch.no_grad { model.parameters.each { |p| p.fill!(0.1) } }
    tokens = Torch.tensor([[0, 1, 2]], dtype: :int64)
    model.call(tokens).narrow(1, 2, 1).sum.backward
    full = model.embedding.weight.grad[0].abs.sum.item
    model.parameters.each { |p| p.grad.zero! if p.grad }
    model.call(tokens, truncate: 1).narrow(1, 2, 1).sum.backward

    assert_operator full, :>, 0
    assert_in_delta 0, model.embedding.weight.grad[0].abs.sum.item, 1e-12
  end
end
