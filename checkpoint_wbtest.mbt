///|
test "checkpoint restores feedback and step count" {
  let sampler = Sampler::new(Config::new(2.0).unwrap(), version=Version::v2())
  let initial = sampler.checkpoint()
  let _ = sampler.sample([2.0, 1.0], 0.5).unwrap()
  assert_eq(sampler.steps(), 1)
  sampler.restore(initial).unwrap()
  assert_eq(sampler.steps(), 0)
  assert_eq(sampler.mu(), initial.mu())
}

///|
test "forks evolve independently" {
  let first = Sampler::new(Config::new(2.0).unwrap())
  let second = first.fork()
  let _ = first.sample([3.0, 2.0, 1.0], 0.5).unwrap()
  assert_eq(first.steps(), 1)
  assert_eq(second.steps(), 0)
}

///|
test "mismatched checkpoint leaves state alone" {
  let first = Sampler::new(Config::new(2.0).unwrap())
  let second = Sampler::new(Config::new(3.0).unwrap())
  let before = second.checkpoint()
  assert_true(second.restore(first.checkpoint()) is Err(_))
  assert_eq(second.checkpoint(), before)
}
