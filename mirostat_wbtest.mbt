///|
test "configuration rejects invalid feedback settings" {
  assert_true(Config::new(0.0) is Err(_))
  assert_true(Config::new(3.0, eta=0.0) is Err(_))
  assert_true(Config::new(3.0, m=1) is Err(_))
  assert_true(Config::new(3.0) is Ok(_))
}

///|
test "V1 consumes one draw and advances state" {
  let config = Config::new(3.0).unwrap()
  let sampler = Sampler::new(config)
  let step = sampler.sample([4.0, 3.0, 2.0, 1.0], 0.4).unwrap()
  assert_true(step.token() >= 0 && step.token() < 4)
  assert_true(step.kept_tokens() >= 1 && step.kept_tokens() <= 4)
  assert_eq(sampler.steps(), 1)
  assert_eq(sampler.mu(), step.mu_after())
  sampler.reset()
  assert_eq(sampler.steps(), 0)
  assert_eq(sampler.mu(), config.initial_mu())
}

///|
test "V2 cuts by surprisal and always retains one token" {
  let config = Config::new(1.0, initial_mu=0.0).unwrap()
  let sampler = Sampler::new(config, version=V2)
  let step = sampler.sample([4.0, 3.0, 2.0], 0.9).unwrap()
  assert_eq(step.kept_tokens(), 1)
  assert_eq(step.token(), 0)
  assert_eq(sampler.version(), V2)
}

///|
test "invalid draw cannot alter sampler state" {
  let config = Config::new(3.0).unwrap()
  let sampler = Sampler::new(config, version=V2)
  let before = sampler.mu()
  assert_eq(sampler.sample([1.0, 0.0], 1.0), Err(InvalidUniform))
  assert_eq(sampler.steps(), 0)
  assert_eq(sampler.mu(), before)
}

///|
test "Zipf exponent is positive for decreasing probabilities" {
  let p = [0.5, 0.25, 0.125, 0.125]
  let s = zipf_exponent(p, [0, 1, 2, 3], 4).unwrap()
  assert_true(s > 0.0)
  assert_true(estimated_k(3.0, s, 4).unwrap() >= 1)
}

///|
test "flat vocabulary remains sampleable in V1" {
  let sampler = Sampler::new(Config::new(2.0).unwrap())
  let step = sampler.sample([0.0, 0.0, 0.0, 0.0], 0.75).unwrap()
  assert_true(step.token() >= 0 && step.token() < 4)
  assert_eq(step.original_probability(), 0.25)
}

///|
test "underflowed tail does not break Zipf estimation" {
  let sampler = Sampler::new(Config::new(2.0).unwrap())
  let step = sampler.sample([1000.0, 999.0, -1000.0], 0.3).unwrap()
  assert_true(step.token() == 0 || step.token() == 1)
}

///|
test "V2 feedback responds to a surprising token" {
  let config = Config::new(1.0, eta=0.5, initial_mu=5.0).unwrap()
  let sampler = Sampler::new(config, version=Version::v2())
  let step = sampler.sample([2.0, 1.0, 0.0], 0.99).unwrap()
  assert_true(step.observed_surprise() > 1.0)
  assert_true(step.mu_after() < step.mu_before())
}

///|
test "one-token distribution is valid for both versions" {
  let config = Config::new(2.0).unwrap()
  let first = Sampler::new(config).sample([5.0], 0.9).unwrap()
  let second = Sampler::new(config, version=Version::v2())
    .sample([5.0], 0.9)
    .unwrap()
  assert_eq(first.token(), 0)
  assert_eq(second.token(), 0)
  assert_eq(first.observed_surprise(), 0.0)
}

///|
test "Zipf power two has exponent two" {
  let estimate = zipf_exponent([1.0, 0.25, 1.0 / 9.0, 0.0625], [0, 1, 2, 3], 4).unwrap()
  assert_true(estimate > 1.999999 && estimate < 2.000001)
  assert_eq(estimated_k(2.0, 2.0, 4), Ok(2))
}

///|
test "V2 cutoff includes tokens exactly at mu" {
  assert_eq(v2_prefix_size([0.5, 0.25, 0.125, 0.125], [0, 1, 2, 3], 2.0), Ok(2))
}

///|
test "weights interface matches logit interface" {
  let config = Config::new(2.0).unwrap()
  let logits = [@math.ln(4.0), @math.ln(2.0), 0.0]
  let from_logits = Sampler::new(config, version=Version::v2())
    .sample(logits, 0.4)
    .unwrap()
  let from_weights = Sampler::new(config, version=Version::v2())
    .sample_weights([4.0, 2.0, 1.0], 0.4)
    .unwrap()
  assert_eq(from_logits.token(), from_weights.token())
  assert_eq(from_logits.kept_tokens(), from_weights.kept_tokens())
}

///|
test "optimized candidate selection matches reference prefix sizes" {
  let logits = [3.0, 2.0, 1.0, 0.0, -1.0, -2.0]
  let p = probabilities(logits).unwrap()
  let order = rank(p).unwrap()
  for mu in [0.0, 1.0, 2.0, 4.0, 8.0] {
    let config = Config::new(2.0, initial_mu=mu).unwrap()
    let v2 = Sampler::new(config, version=Version::v2())
    assert_eq(
      v2.preview(logits).unwrap().kept_tokens(),
      v2_prefix_size(p, order, mu).unwrap(),
    )
    let v1 = Sampler::new(config)
    let exponent = zipf_exponent(p, order, config.m()).unwrap()
    assert_eq(
      v1.preview(logits).unwrap().kept_tokens(),
      estimated_k(mu, exponent, 6).unwrap(),
    )
  }
}
