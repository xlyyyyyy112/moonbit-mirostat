///|
/// Two independent sessions in one inference batch. The second call uses an
/// invalid row to show that atomic batching restores both feedback states.
fn main {
  let config = @mirostat.Config::new(2.0).unwrap()
  let first = @mirostat.Sampler::new(config)
  let second = @mirostat.Sampler::new(config, version=@mirostat.Version::v2())
  let samplers = [first, second]
  let decisions = @mirostat.sample_batch_atomic(
    samplers,
    [[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]],
    [0.25, 0.75],
  ).unwrap()
  for index in 0..<decisions.length() {
    println(
      "session=" +
      index.to_string() +
      " token=" +
      decisions[index].token().to_string() +
      " mu=" +
      samplers[index].mu().to_string(),
    )
  }
  let before_first = first.mu()
  let before_second = second.mu()
  let failed = @mirostat.sample_batch_atomic(samplers, [[3.0, 2.0, 1.0], []], [
    0.25, 0.75,
  ])
  match failed {
    Err(error) => println("rollback: " + error.message())
    Ok(_) => println("unexpected success")
  }
  println(
    "state_restored=" +
    (first.mu() == before_first && second.mu() == before_second).to_string(),
  )
}
