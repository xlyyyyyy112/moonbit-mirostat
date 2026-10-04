///|
test "batch maintains separate Mirostat feedback states" {
  let config = Config::new(2.0).unwrap()
  let first = Sampler::new(config)
  let second = Sampler::new(config, version=Version::v2())
  let output = sample_batch(
    [first, second],
    [[3.0, 2.0, 1.0], [0.0, 0.0, 0.0]],
    [0.1, 0.9],
  ).unwrap()
  assert_eq(output.length(), 2)
  assert_eq(first.steps(), 1)
  assert_eq(second.steps(), 1)
  assert_true(first.mu() != second.mu())
}

///|
test "atomic batch rolls back all sessions" {
  let config = Config::new(2.0).unwrap()
  let first = Sampler::new(config)
  let second = Sampler::new(config)
  let before_first = first.checkpoint()
  let before_second = second.checkpoint()
  let result = sample_batch_atomic([first, second], [[1.0, 0.0], []], [0.5, 0.5])
  assert_true(result is Err(_))
  assert_eq(first.checkpoint(), before_first)
  assert_eq(second.checkpoint(), before_second)
}

///|
test "mismatched batch input fails before sampling" {
  let sampler = Sampler::new(Config::new(2.0).unwrap())
  assert_true(sample_batch([sampler], [], [0.5]) is Err(_))
  assert_eq(sampler.steps(), 0)
}

///|
test "non-atomic batch retains earlier successful session" {
  let config = Config::new(2.0).unwrap()
  let first = Sampler::new(config)
  let second = Sampler::new(config)
  let result = sample_batch([first, second], [[1.0], []], [0.2, 0.2])
  assert_true(result is Err(EmptyLogits))
  assert_eq(first.steps(), 1)
  assert_eq(second.steps(), 0)
}

///|
test "shared prepared batch preserves separate feedback" {
  let row = PreparedRow::from_logits([3.0, 2.0, 1.0]).unwrap()
  let config = Config::new(2.0).unwrap()
  let first = Sampler::new(config)
  let second = Sampler::new(config, version=Version::v2())
  let steps = sample_shared_prepared([first, second], row, [0.1, 0.9]).unwrap()
  assert_eq(steps.length(), 2)
  assert_eq(first.steps(), 1)
  assert_eq(second.steps(), 1)
  assert_true(sample_shared_prepared([first], row, []) is Err(_))
}
