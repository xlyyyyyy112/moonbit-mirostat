///|
/// A model backend can hand over positive weights instead of logits.
fn main {
  let sampler = @mirostat.Sampler::new(
    @mirostat.Config::new(2.0).unwrap(),
    version=@mirostat.Version::v2(),
  )
  let weights = [8.0, 4.0, 2.0, 1.0]
  let preview = sampler.preview_weights(weights).unwrap()
  println("vocabulary=" + preview.vocabulary().to_string())
  println("candidates=" + preview.kept_tokens().to_string())
  println("retained_mass=" + preview.retained_mass().to_string())
  let chosen = sampler.sample_weights(weights, 0.7).unwrap()
  println("token=" + chosen.token().to_string())
  println("surprise_bits=" + chosen.observed_surprise().to_string())
}
