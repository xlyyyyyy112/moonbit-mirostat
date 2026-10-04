///|
test "raw and prepared paths agree over varied vocabularies" {
  for size in [1, 2, 3, 16, 127] {
    let logits : Array[Double] = []
    for index in 0..<size {
      let value = ((index * 37 + 11) % 53).to_double() / 7.0
      logits.push(if index > 0 && index % 7 == 0 { -1.0 / 0.0 } else { value })
    }
    let row = PreparedRow::from_logits(logits).unwrap()
    for version in [Version::v1(), Version::v2()] {
      let first = Sampler::new(Config::new(2.5).unwrap(), version~)
      let second = Sampler::new(Config::new(2.5).unwrap(), version~)
      for draw in [0.0, 0.1, 0.49, 0.9, 0.999999] {
        let raw = first.sample(logits, draw).unwrap()
        let prepared = second.sample_prepared(row, draw).unwrap()
        assert_eq(raw, prepared)
        assert_true(raw.token() >= 0 && raw.token() < size)
        assert_true(raw.original_probability() > 0.0)
        assert_true(raw.kept_tokens() >= 1 && raw.kept_tokens() <= size)
      }
    }
  }
}

///|
test "V2 retained set grows monotonically with mu" {
  let p = probabilities([4.0, 3.0, 2.0, 1.0, 0.0]).unwrap()
  let order = rank(p).unwrap()
  let mut prior = 0
  for mu in [0.0, 0.5, 1.0, 2.0, 3.0, 8.0] {
    let size = v2_prefix_size(p, order, mu).unwrap()
    assert_true(size >= prior)
    prior = size
  }
}

///|
test "top-p retained set grows monotonically with threshold" {
  let p = probabilities([4.0, 3.0, 2.0, 1.0, 0.0]).unwrap()
  let order = rank(p).unwrap()
  let mut prior = 0
  for threshold in [0.1, 0.3, 0.5, 0.8, 1.0] {
    let size = top_p_prefix_size(p, order, threshold).unwrap()
    assert_true(size >= prior)
    prior = size
  }
}

///|
test "reset reproduces a run with identical random stream" {
  let sampler = Sampler::new(Config::new(2.5).unwrap())
  let first_rng = ParkMiller::new(17).unwrap()
  let first : Array[Int] = []
  for _ in 0..<20 {
    first.push(
      sampler.sample_with_rng([3.0, 2.0, 1.0], first_rng).unwrap().token(),
    )
  }
  sampler.reset()
  let second_rng = ParkMiller::new(17).unwrap()
  let second : Array[Int] = []
  for _ in 0..<20 {
    second.push(
      sampler.sample_with_rng([3.0, 2.0, 1.0], second_rng).unwrap().token(),
    )
  }
  assert_eq(first, second)
}

///|
test "very large finite logits do not overflow softmax" {
  let logits = [1.0e300, 1.0e300, -1.0e300]
  let p = probabilities(logits).unwrap()
  assert_eq(p, [0.5, 0.5, 0.0])
  let sampler = Sampler::new(Config::new(1.0).unwrap())
  assert_true(sampler.sample(logits, 0.99) is Ok(_))
}
