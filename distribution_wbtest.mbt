///|
test "uniform entropy equals two bits" {
  assert_eq(entropy([0.25, 0.25, 0.25, 0.25]), Ok(2.0))
  assert_eq(perplexity(2.0), Ok(4.0))
}

///|
test "prefix expectation reflects truncation" {
  let p = [0.5, 0.25, 0.25]
  let order = [0, 1, 2]
  assert_eq(prefix_mass(p, order, 1), Ok(0.5))
  assert_eq(expected_prefix_surprise(p, order, 1), Ok(1.0))
  assert_true(expected_prefix_surprise(p, order, 3).unwrap() > 1.0)
}

///|
test "distribution diagnostics reject malformed inputs" {
  assert_true(entropy([]) is Err(_))
  assert_true(entropy([1.0, -1.0]) is Err(_))
  assert_true(prefix_mass([0.5, 0.5], [0, 0], 1) is Err(_))
  assert_true(perplexity(-1.0) is Err(_))
}

///|
test "truncation entropy and KL have expected limits" {
  let p = [0.5, 0.25, 0.25]
  let order = [0, 1, 2]
  assert_eq(prefix_entropy(p, order, 1), Ok(0.0))
  assert_eq(prefix_kl(p, order, 1), Ok(1.0))
  assert_eq(prefix_kl(p, order, 3), Ok(0.0))
  assert_eq(prefix_surprise_variance(p, order, 1), Ok(0.0))
  assert_true(prefix_surprise_variance(p, order, 3).unwrap() > 0.0)
}
