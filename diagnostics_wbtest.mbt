///|
test "replay and summary expose adaptive behavior" {
  let sampler = Sampler::new(Config::new(2.0).unwrap(), version=Version::v2())
  let trace = replay(
    sampler,
    [[3.0, 2.0, 1.0], [2.0, 1.0, 0.0], [1.0, 0.5, 0.0]],
    [0.2, 0.7, 0.1],
  ).unwrap()
  assert_eq(trace.length(), 3)
  assert_eq(sampler.steps(), 3)
  let summary = summarize(trace, 2.0).unwrap()
  assert_eq(summary.count(), 3)
  assert_true(summary.mean_kept() >= 1.0)
  assert_eq(summary.final_mu(), sampler.mu())
}

///|
test "replay rejects mismatched rows before state changes" {
  let sampler = Sampler::new(Config::new(2.0).unwrap())
  assert_true(replay(sampler, [[0.0]], []) is Err(_))
  assert_eq(sampler.steps(), 0)
}

///|
test "replay stops after a malformed row" {
  let sampler = Sampler::new(Config::new(2.0).unwrap())
  assert_true(replay(sampler, [[1.0], []], [0.1, 0.2]) is Err(_))
  assert_eq(sampler.steps(), 1)
}

///|
test "running and rolling surprise cover each prefix" {
  let sampler = Sampler::new(Config::new(2.0).unwrap(), version=Version::v2())
  let trace = replay(sampler, [[3.0, 2.0], [2.0, 1.0], [1.0, 0.0]], [
    0.1, 0.9, 0.5,
  ]).unwrap()
  let running = running_surprise(trace)
  let rolling = rolling_surprise(trace, 2).unwrap()
  assert_eq(running.length(), 3)
  assert_eq(rolling.length(), 3)
  assert_eq(running[0], rolling[0])
  assert_true(rolling_surprise(trace, 0) is Err(_))
}

///|
test "atomic replay rolls back a partially valid batch" {
  let sampler = Sampler::new(Config::new(2.0).unwrap())
  let before = sampler.checkpoint()
  assert_true(replay_atomic(sampler, [[1.0], []], [0.1, 0.2]) is Err(_))
  assert_eq(sampler.checkpoint(), before)
}
