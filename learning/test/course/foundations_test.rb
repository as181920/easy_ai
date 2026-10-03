require_relative "../test_helper"

class FoundationsTest < Minitest::Test
  def test_stable_probabilities_and_cross_entropy
    math = EasyAILearning::Foundations::Math

    assert_in_delta 1, math.softmax([1000, 1001]).sum, 1e-12
    assert_in_delta Math.log(2), math.cross_entropy([1000, 1000], 1), 1e-12
    assert_in_delta 0, math.sigmoid(-1000), 1e-12
    assert_in_delta 1, math.sigmoid(1000), 1e-12
    assert_equal [[1, 1], [0, 1]], math.confusion([0, 1, 1], [0, 0, 1])[:matrix]
  end

  def test_linear_and_logistic_derivatives_against_finite_differences
    [false, true].each do |logistic|
      model = EasyAILearning::Foundations::Linear.new(features: 2, logistic: logistic)
      rows, targets = [[0.2, 0.4], [-0.1, 0.3]], [0.0, 1.0]
      g = model.gradients(rows, targets)
      model.weights.each_index do |i|
        numerical = EasyAILearning::Foundations::Math.finite_difference(0) do |value|
          model.weights[i] = value
          model.loss(rows, targets)
        end
        model.weights[i] = 0

        assert_in_delta g[:weights][i], numerical, 1e-8
      end
    end
  end

  def test_pca_known_axis_and_reconstruction
    pca = EasyAILearning::Foundations::Pca.new.fit([[-2, 0], [0, 0], [2, 0]])

    assert_in_delta 1, pca.components.first.first.abs, 1e-9
    assert_in_delta 8.0 / 3, pca.eigenvalues.first, 1e-9
    actual = pca.reconstruct(pca.transform([[1, 0]])).first

    assert_in_delta 1, actual[0], 1e-9
    assert_in_delta 0, actual[1], 1e-9
  end

  def test_tree_split_and_ensemble_mean_are_exact
    rows, labels = [[0], [1], [2], [3]], [0, 0, 1, 1]
    tree = EasyAILearning::Foundations::Tree.new.fit(rows, labels, depth: 1)

    assert_in_delta 1.5, tree.root[:threshold], 1e-12
    assert_equal labels, rows.map { |r| tree.predict(r) }
    ensemble = EasyAILearning::Foundations::Ensemble.new.fit(rows, labels, count: 3)
    expected = ensemble.trees.sum { |t| t.predict([1]) } / 3.0

    assert_in_delta expected, ensemble.predict([1]), 1e-12
  end

  def test_kmeans_preserves_cluster_means
    model = EasyAILearning::Foundations::KMeans.new.fit([[-2], [-1], [1], [2]], seed: 2)

    assert_equal [-1.5, 1.5], model.centers.flatten.sort
  end
end

class FoundationEdgeCasesTest < Minitest::Test
  def test_rank_deficient_pca_keeps_orthonormal_basis
    model = EasyAILearning::Foundations::Pca.new.fit([[-2, 0, 0], [0, 0, 0], [2, 0, 0]], dimensions: 3)

    model.components.each do |a|
      assert_in_delta 1, EasyAILearning::Foundations::Math.dot(a, a), 1e-9
    end
    assert_in_delta 0, EasyAILearning::Foundations::Math.dot(model.components[0], model.components[1]), 1e-9
    assert_in_delta 0, EasyAILearning::Foundations::Math.dot(model.components[1], model.components[2]), 1e-9
  end

  def test_data_generators_are_reproducible_and_do_not_consume_global_rng
    data = EasyAILearning::Course::Data
    first = data.images(count: 2, seed: 9)
    srand(17)
    expected = rand
    srand(17)
    second = data.images(count: 2, seed: 9)

    assert_equal first, second
    assert_equal expected, rand
    refute_equal first, data.images(count: 2, seed: 10)
  end

  def test_histogram_count_and_singular_spectrum
    stats = EasyAILearning::Diagnostics::Stats
    histogram = stats.histogram([-1.0, 0.0, 0.5, 1.0], bins: 2)
    spectrum = stats.spectrum(Torch.tensor([[2.0, 0.0], [0.0, 0.0]]))

    assert_equal 4, histogram.sum(&:last)
    assert_equal [2.0, 0.0], spectrum[:singular_values]
    assert_in_delta 1, spectrum[:effective_rank], 1e-12
  end
end
