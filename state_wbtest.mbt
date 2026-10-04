///|
test "portable state resumes an equivalent session" {
  let config = Config::new(2.0).unwrap()
  let original = Sampler::new(config, version=Version::v2())
  let _ = original.sample([3.0, 2.0, 1.0], 0.4).unwrap()
  let resumed = Sampler::from_state(config, Version::v2(), original.state())
  assert_eq(
    original.sample([3.0, 2.0, 1.0], 0.8),
    resumed.sample([3.0, 2.0, 1.0], 0.8),
  )
  assert_eq(original.state(), resumed.state())
}

///|
test "portable state rejects invalid counters" {
  assert_true(SamplerState::new(0.0 / 0.0, 0) is Err(_))
  assert_true(SamplerState::new(1.0, -1) is Err(_))
}
