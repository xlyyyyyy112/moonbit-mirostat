///|
test "generation callback sees evolving token prefix" {
  let sampler = Sampler::new(Config::new(2.0).unwrap(), version=Version::v2())
  let rng = ParkMiller::new(42).unwrap()
  let output = generate(
    sampler,
    rng,
    fn(prefix) {
      if prefix.length() == 0 {
        Ok([5.0, 0.0])
      } else {
        Ok([0.0, 5.0])
      }
    },
    5,
    Some(1),
  ).unwrap()
  assert_eq(output.tokens(), [0, 1])
  assert_eq(output.steps().length(), 2)
  assert_eq(output.reason(), StopReason::end_token())
  assert_eq(sampler.steps(), 2)
}

///|
test "generation with zero limit does not call model or RNG" {
  let sampler = Sampler::new(Config::new(2.0).unwrap())
  let rng = ParkMiller::new(42).unwrap()
  let before = rng.state()
  let output = generate(sampler, rng, fn(_) { Err(EmptyLogits) }, 0, None).unwrap()
  assert_eq(output.reason(), StopReason::limit())
  assert_eq(output.tokens().length(), 0)
  assert_eq(rng.state(), before)
}

///|
test "generation stops before sampling a failed model row" {
  let sampler = Sampler::new(Config::new(2.0).unwrap())
  let rng = ParkMiller::new(42).unwrap()
  let result = generate(
    sampler,
    rng,
    fn(prefix) {
      if prefix.length() == 0 {
        Ok([1.0])
      } else {
        Err(EmptyLogits)
      }
    },
    3,
    None,
  )
  assert_true(result is Err(EmptyLogits))
  assert_eq(sampler.steps(), 1)
}

///|
test "prompt is visible to model but excluded from output" {
  let sampler = Sampler::new(Config::new(2.0).unwrap(), version=Version::v2())
  let output = generate_from_prompt(
    sampler,
    ParkMiller::new(4).unwrap(),
    fn(context) {
      if context.length() == 2 {
        Ok([5.0, 0.0])
      } else {
        Ok([0.0, 5.0])
      }
    },
    [7, 8],
    3,
    Some(1),
  ).unwrap()
  assert_eq(output.tokens(), [0, 1])
  assert_true(
    generate_from_prompt(
      sampler,
      ParkMiller::new(4).unwrap(),
      fn(_) { Ok([1.0]) },
      [-1],
      1,
      None,
    )
    is Err(_),
  )
}

///|
test "stream observer can stop after a successful token" {
  let sampler = Sampler::new(Config::new(2.0).unwrap(), version=Version::v2())
  let mut observed = 0
  let output = generate_stream(
    sampler,
    ParkMiller::new(3).unwrap(),
    fn(_) { Ok([2.0, 1.0]) },
    [9],
    10,
    None,
    fn(_) {
      observed = observed + 1
      observed < 3
    },
  ).unwrap()
  assert_eq(output.reason(), StopReason::observer())
  assert_eq(output.count(), 3)
  assert_eq(sampler.steps(), 3)
}
