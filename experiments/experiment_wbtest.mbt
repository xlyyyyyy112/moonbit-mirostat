///|
test "Zipf fixture decreases by rank" {
  let logits = zipf_logits(4, 1.5)
  assert_eq(logits.length(), 4)
  assert_true(logits[0] > logits[1] && logits[1] > logits[2])
}

///|
test "static experiment produces repeatable trace" {
  let config = @mirostat.Config::new(3.0).unwrap()
  let first = run_static(
    @mirostat.Sampler::new(config),
    @mirostat.ParkMiller::new(42).unwrap(),
    32,
    1.2,
    20,
  ).unwrap()
  let second = run_static(
    @mirostat.Sampler::new(config),
    @mirostat.ParkMiller::new(42).unwrap(),
    32,
    1.2,
    20,
  ).unwrap()
  assert_eq(first, second)
}

///|
test "alternating experiment changes model rows" {
  let config = @mirostat.Config::new(3.0).unwrap()
  let sampler = @mirostat.Sampler::new(config, version=@mirostat.Version::v2())
  let trace = run_alternating(
    sampler,
    @mirostat.ParkMiller::new(7).unwrap(),
    16,
    2.0,
    0.5,
    10,
  ).unwrap()
  assert_eq(trace.length(), 10)
  assert_eq(sampler.steps(), 10)
}

///|
test "long static run approaches requested surprise" {
  let config = @mirostat.Config::new(3.0).unwrap()
  let sampler = @mirostat.Sampler::new(config, version=@mirostat.Version::v2())
  let trace = run_static(
    sampler,
    @mirostat.ParkMiller::new(42).unwrap(),
    256,
    1.15,
    1000,
  ).unwrap()
  let result = @mirostat.summarize(trace, config.tau()).unwrap()
  assert_true(result.mean_surprise() > 2.8)
  assert_true(result.mean_surprise() < 3.2)
}
