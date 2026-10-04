///|
test "online summary matches replay summary" {
  let sampler = Sampler::new(Config::new(2.0).unwrap(), version=Version::v2())
  let steps = replay(sampler, [[2.0, 1.0], [1.0, 0.0], [0.0, -1.0]], [
    0.2, 0.8, 0.4,
  ]).unwrap()
  let online = TraceAccumulator::new(2.0).unwrap()
  for step in steps {
    online.push(step).unwrap()
  }
  assert_eq(online.count(), 3)
  assert_eq(online.summary(), summarize(steps, 2.0))
}

///|
test "online summary requires observations" {
  let online = TraceAccumulator::new(2.0).unwrap()
  assert_true(online.summary() is Err(_))
}
