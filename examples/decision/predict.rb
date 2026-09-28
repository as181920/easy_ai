require "bundler/setup"
$LOAD_PATH.unshift File.expand_path("../../lib", __dir__)
require "easy_ai"
require "json"

checkpoint = ARGV.fetch(0) { abort "Usage: ruby examples/decision/predict.rb CHECKPOINT_OR_RUN" }
predictor = EasyAI::Decision::Predictor.load(checkpoint)
result = predictor.probabilities(
  state: "我的订单被重复扣款了。",
  question: "应该交给哪个部门处理？",
  options: [
    { id: "billing", text: "处理账单、支付和退款问题" },
    { id: "shipping", text: "处理物流和配送问题" },
    { id: "account", text: "处理账号和登录问题" }
  ]
)
puts JSON.pretty_generate(result)
