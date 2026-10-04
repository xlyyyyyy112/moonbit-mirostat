///|
test "external candidate weights normalize and preserve state" {
  let sampler = Sampler::new(Config::new(2.0).unwrap(), version=Version::v2())
  let before = sampler.state()
  let candidates = sampler.candidates([3.0, 2.0, 1.0]).unwrap()
  let tokens = candidates.tokens()
  let conditional = candidates.conditional()
  let original = candidates.original()
  assert_eq(tokens.length(), conditional.length())
  assert_eq(tokens.length(), original.length())
  let mut total = 0.0
  for value in conditional {
    total = total + value
  }
  assert_true(total > 0.999999 && total < 1.000001)
  assert_eq(sampler.state(), before)
  assert_true(sampler.observe_token([3.0, 2.0, 1.0], tokens[0]) is Ok(_))
}

///|
test "candidate lookup uses vocabulary token ids" {
  let sampler = Sampler::new(
    Config::new(1.0, initial_mu=0.0).unwrap(),
    version=Version::v2(),
  )
  let set = sampler.candidates([4.0, 3.0, 2.0]).unwrap()
  assert_eq(set.position(0), Some(0))
  assert_eq(set.position(1), None)
  assert_eq(set.conditional_probability(0), Some(1.0))
  assert_true(set.original_probability(0) is Some(_))
  assert_eq(set.original_probability(2), None)
}
