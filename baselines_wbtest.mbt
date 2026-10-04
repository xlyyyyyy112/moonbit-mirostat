///|
test "top-k one is greedy and stable under ties" {
  assert_eq(sample_top_k([2.0, 3.0, 1.0], 1, 0.99), Ok(1))
  assert_eq(sample_top_k([2.0, 2.0], 1, 0.99), Ok(0))
}

///|
test "top-p uses smallest sufficient prefix" {
  let p = probabilities([3.0, 2.0, 1.0]).unwrap()
  let order = rank(p).unwrap()
  assert_eq(top_p_prefix_size(p, order, 0.5), Ok(1))
  assert_eq(top_p_prefix_size(p, order, 1.0), Ok(3))
}

///|
test "temperature one preserves model probabilities" {
  let logits = [4.0, 3.0, 1.0]
  let scaled = with_temperature(logits, 1.0).unwrap()
  let original = probabilities(logits).unwrap()
  let after = probabilities(scaled).unwrap()
  for i in 0..<3 {
    let difference = original[i] - after[i]
    assert_true(difference < 0.000000001 && difference > -0.000000001)
  }
}

///|
test "baselines reject invalid parameters" {
  assert_true(sample_top_k([0.0], 0, 0.5) is Err(_))
  assert_true(sample_top_p([0.0], 0.0, 0.5) is Err(_))
  assert_true(with_temperature([0.0], 0.0) is Err(_))
  assert_true(sample_temperature([], 1.0, 0.5) is Err(_))
}
