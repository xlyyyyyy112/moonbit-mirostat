///|
/// Reproducible five-step demonstration. The caller supplies independent
/// uniform draws, as it would from its own random generator in production.
fn main {
  let config = @mirostat.Config::new(2.5).unwrap()
  let v1 = @mirostat.Sampler::new(config)
  let v2 = @mirostat.Sampler::new(config, version=@mirostat.Version::v2())
  let draws = [0.10, 0.75, 0.42, 0.91, 0.25]
  for draw in draws {
    let logits = [4.0, 3.1, 2.4, 1.8, 1.0, 0.2]
    let first = v1.sample(logits, draw).unwrap()
    let second = v2.sample(logits, draw).unwrap()
    println(
      "v1 token=" +
      first.token().to_string() +
      " keep=" +
      first.kept_tokens().to_string() +
      " mu=" +
      first.mu_after().to_string(),
    )
    println(
      "v2 token=" +
      second.token().to_string() +
      " keep=" +
      second.kept_tokens().to_string() +
      " mu=" +
      second.mu_after().to_string(),
    )
  }
}
