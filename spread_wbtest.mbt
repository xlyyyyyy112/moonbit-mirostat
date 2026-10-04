///|
test "spread bounds contain all trace surprises" {
  let sampler = Sampler::new(Config::new(2.0).unwrap())
  let trace = replay(
    sampler,
    [[3.0, 2.0, 1.0], [2.0, 1.0, 0.0], [1.0, 0.5, 0.0]],
    [0.1, 0.8, 0.4],
  ).unwrap()
  let spread = trace_spread(trace).unwrap()
  assert_true(spread.minimum() <= spread.median())
  assert_true(spread.median() <= spread.maximum())
  assert_true(spread.variance() >= 0.0)
  assert_true(spread.minimum_kept() <= spread.maximum_kept())
  assert_true(trace_spread([]) is Err(_))
}
