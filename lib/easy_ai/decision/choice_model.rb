module EasyAI
  module Decision
    class ChoiceModel < Torch::NN::Module
      attr_reader :config, :encoder

      def initialize(config = Config.new)
        super()
        @config = config
        @encoder = Encoder.new(config)
        separate = config[:model]["encoding_mode"] == "separate"
        @interactions = Torch::NN::ModuleList.new(Array.new(separate ? config[:model]["interaction_layers"] : 0) { InteractionBlock.new(config) })
        @score = Torch::NN::Linear.new(config[:model]["hidden_size"], 1)
        @matching = Torch::NN::Linear.new(config[:model]["hidden_size"] * 4, config[:model]["hidden_size"]) if separate && config[:model]["score_mode"] == "matching"
        @evidence_head = EvidenceHead.new(config[:model]["hidden_size"]) if config[:model]["evidence_head"]
        # Initialize tied embedding at a scale suitable for MLM logits.
        Torch.no_grad do
          @encoder.embedding.weight.normal!(mean: 0.0, std: 0.02)
          @encoder.embedding.weight[0].zero!
        end
      end

      def forward(batch)
        return score_joint(batch) if config[:model]["encoding_mode"] == "joint"
        memory = encode_state(batch.fetch(:state_ids), batch.fetch(:state_mask))
        score_candidates(memory, batch.fetch(:state_mask), batch.fetch(:option_ids), batch.fetch(:option_mask), batch.fetch(:candidate_mask),
          answer_mask: batch[:answer_mask])
      end

      def score_joint(batch)
        b, k, length = batch.fetch(:joint_ids).shape
        mask = batch.fetch(:joint_mask).view([b * k, length])
        hidden = encoder.call(batch.fetch(:joint_ids).view([b * k, length]), mask: mask)
        pool_mask = config[:model]["pooling"] == "candidate" ? batch.fetch(:joint_answer_mask).view([b * k, length]) : mask
        weights = pool_mask.to(dtype: :float32).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim: 1) / weights.sum(dim: 1).clamp(min: 1.0)
        @score.call(pooled).view([b, k]).masked_fill(Torch.logical_not(batch.fetch(:candidate_mask)), -Float::INFINITY)
      end

      def forward_with_evidence(batch)
        raise ArgumentError, "Model has no evidence head" unless @evidence_head
        memory = encode_state(batch.fetch(:state_ids), batch.fetch(:state_mask))
        score_candidates(memory, batch.fetch(:state_mask), batch.fetch(:option_ids), batch.fetch(:option_mask), batch.fetch(:candidate_mask),
          answer_mask: batch[:answer_mask], sentence_mask: batch.fetch(:sentence_mask))
      end

      def encode_state(ids, mask)
        encoder.call(ids, mask: mask)
      end

      def score_candidates(memory, memory_mask, option_ids, option_mask, candidate_mask, answer_mask: nil, sentence_mask: nil)
        b, k, m = option_ids.shape
        d = config[:model]["hidden_size"]
        query_mask = option_mask.view([b * k, m])
        query = encoder.call(option_ids.view([b * k, m]), mask: query_mask)
        states = memory.unsqueeze(1).expand(b, k, memory.shape[1], d).contiguous.view([b * k, memory.shape[1], d])
        states_mask = memory_mask.unsqueeze(1).expand(b, k, memory_mask.shape[1]).contiguous.view([b * k, memory_mask.shape[1]])
        @interactions.each { |block| query = block.call(query, memory: states, memory_mask: states_mask) }
        pooling_mask = if config[:model]["pooling"] == "candidate"
          raise ArgumentError, "Candidate pooling requires answer_mask" unless answer_mask
          answer_mask.view([b * k, m])
                       else
          query_mask
                       end
        weights = pooling_mask.to(dtype: :float32).unsqueeze(-1)
        pooled = (query * weights).sum(dim: 1) / weights.sum(dim: 1).clamp(min: 1.0)
        evidence = @evidence_head.call(pooled, memory: memory, sentence_mask: sentence_mask, candidate_mask: candidate_mask) if sentence_mask
        if @matching
          state_weights = states_mask.to(dtype: :float32).unsqueeze(-1)
          state_pooled = (states * state_weights).sum(dim: 1) / state_weights.sum(dim: 1).clamp(min: 1.0)
          features = Torch.cat([pooled, state_pooled, pooled * state_pooled, (pooled - state_pooled).abs], dim: -1)
          pooled = Torch::NN::Functional.gelu(@matching.call(features))
        end
        logits = @score.call(pooled).view([b, k]).masked_fill(Torch.logical_not(candidate_mask), -Float::INFINITY)
        sentence_mask ? { logits: logits, evidence_logits: evidence } : logits
      end

      def mlm_logits(ids, mask:, positions:)
        hidden = encoder.call(ids, mask: mask).view([-1, config[:model]["hidden_size"]])
        selected = hidden.index_select(0, positions)
        Torch.matmul(selected, encoder.embedding.weight.transpose(0, 1))
      end

      def parameter_count
        parameters.sum(&:numel)
      end
    end
  end
end
