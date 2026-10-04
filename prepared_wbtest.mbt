///|
test "prepared row matches repeated raw sampling" {
  let config = Config::new(2.0).unwrap()
  let logits = [4.0, 3.0, 2.0, 1.0]
  let prepared = PreparedRow::from_logits(logits).unwrap()
  for version in [Version::v1(), Version::v2()] {
    let raw = Sampler::new(config, version~)
    let cached = Sampler::new(config, version~)
    for draw in [0.1, 0.9, 0.25, 0.75] {
      assert_eq(
        raw.sample(logits, draw),
        cached.sample_prepared(prepared, draw),
      )
    }
  }
}

///|
test "prepared row owns normalized probabilities" {
  let weights = [4.0, 2.0]
  let row = PreparedRow::from_weights(weights).unwrap()
  weights[0] = 0.0
  assert_true(row.probability(0).unwrap() > 0.6)
  assert_eq(row.vocabulary(), 2)
  assert_true(row.probability(2) is Err(_))
}

///|
test "prepared preview matches ordinary preview" {
  let sampler = Sampler::new(Config::new(2.0).unwrap(), version=Version::v2())
  let logits = [3.0, 2.0, 1.0]
  let row = PreparedRow::from_logits(logits).unwrap()
  let fresh = sampler.preview(logits).unwrap()
  let cached = sampler.preview_prepared(row).unwrap()
  assert_eq(fresh.kept_tokens(), cached.kept_tokens())
  assert_eq(fresh.retained_mass(), cached.retained_mass())
  assert_eq(fresh.expected_surprise(), cached.expected_surprise())
}

///|
test "invalid prepared draw preserves sampler state" {
  let sampler = Sampler::new(Config::new(2.0).unwrap())
  let row = PreparedRow::from_logits([2.0, 1.0]).unwrap()
  let before = sampler.checkpoint()
  assert_eq(sampler.sample_prepared(row, 1.0), Err(InvalidUniform))
  assert_eq(sampler.checkpoint(), before)
}
