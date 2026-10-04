///|
test "preview agrees with next sampled prefix" {
  let sampler = Sampler::new(Config::new(2.0).unwrap(), version=Version::v2())
  let logits = [3.0, 2.0, 1.0]
  let preview = sampler.preview(logits).unwrap()
  assert_eq(preview.vocabulary(), 3)
  assert_eq(sampler.steps(), 0)
  let step = sampler.sample(logits, 0.4).unwrap()
  assert_eq(preview.kept_tokens(), step.kept_tokens())
  assert_true(preview.retained_mass() > 0.0)
  assert_true(preview.expected_surprise() >= 0.0)
}

///|
test "preview weights does not mutate state" {
  let sampler = Sampler::new(Config::new(2.0).unwrap())
  let before = sampler.mu()
  let preview = sampler.preview_weights([3.0, 2.0, 1.0]).unwrap()
  assert_eq(preview.mu(), before)
  assert_eq(sampler.mu(), before)
  assert_eq(sampler.steps(), 0)
}
