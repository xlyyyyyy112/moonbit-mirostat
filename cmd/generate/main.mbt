///|
/// A tiny context-dependent model fixture. The four token ids stand for
/// START, moon, bit, and END; this is an API example, not a trained model.
fn main {
  let sampler = @mirostat.Sampler::new(
    @mirostat.Config::new(1.5).unwrap(),
    version=@mirostat.Version::v2(),
  )
  let rng = @mirostat.ParkMiller::new(9).unwrap()
  let output = @mirostat.generate(
    sampler,
    rng,
    fn(prefix) {
      if prefix.length() == 0 {
        Ok([0.0, 5.0, 0.0, -5.0])
      } else if prefix[prefix.length() - 1] == 1 {
        Ok([0.0, -2.0, 5.0, -5.0])
      } else {
        Ok([-5.0, -5.0, -5.0, 5.0])
      }
    },
    8,
    Some(3),
  ).unwrap()
  for token in output.tokens() {
    let word = match token {
      1 => "moon"
      2 => "bit"
      3 => "<end>"
      _ => "<start>"
    }
    println(word)
  }
  println("steps=" + sampler.steps().to_string())
}
