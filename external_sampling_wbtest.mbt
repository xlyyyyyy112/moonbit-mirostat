///|
test "external token observation matches a drawn token" {
  let config = Config::new(2.0).unwrap()
  let drawn = Sampler::new(config, version=Version::v2())
  let observed = Sampler::new(config, version=Version::v2())
  let logits = [3.0, 2.0, 1.0]
  let step = drawn.sample(logits, 0.25).unwrap()
  assert_eq(step, observed.observe_token(logits, step.token()).unwrap())
  assert_eq(drawn.state(), observed.state())
}

///|
test "external observation cannot force a filtered token" {
  let sampler = Sampler::new(
    Config::new(1.0, initial_mu=0.0).unwrap(),
    version=Version::v2(),
  )
  let before = sampler.state()
  assert_true(sampler.observe_token([4.0, 3.0, 2.0], 2) is Err(_))
  assert_eq(sampler.state(), before)
}

///|
test "prepared observation matches ordinary observation" {
  let config = Config::new(2.0).unwrap()
  let ordinary = Sampler::new(config)
  let prepared = Sampler::new(config)
  let logits = [3.0, 2.0, 1.0]
  let row = PreparedRow::from_logits(logits).unwrap()
  assert_eq(
    ordinary.observe_token(logits, 0),
    prepared.observe_prepared_token(row, 0),
  )
}
