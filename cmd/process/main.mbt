///|
/// Explicit logit preprocessing before adaptive selection. The example
/// applies history costs and a grammar-like token allow-mask.
fn main {
  let raw = [3.0, 2.0, 1.0, 0.0]
  let history = [0, 0, 1]
  let penalized = @mirostat.penalize_counts(raw, history, 0.5, 0.25).unwrap()
  let allowed = @mirostat.mask_logits(penalized, [false, true, true, false]).unwrap()
  let sampler = @mirostat.Sampler::new(
    @mirostat.Config::new(1.5).unwrap(),
    version=@mirostat.Version::v2(),
  )
  let step = sampler.sample(allowed, 0.6).unwrap()
  println("token=" + step.token().to_string())
  println("kept=" + step.kept_tokens().to_string())
  println("mu_after=" + step.mu_after().to_string())
}
