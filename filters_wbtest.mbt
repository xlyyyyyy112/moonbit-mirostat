///|
test "min-p is relative to the highest probability" {
  let p = [0.5, 0.3, 0.2]
  assert_eq(min_p_prefix_size(p, [0, 1, 2], 0.5), Ok(2))
  assert_eq(min_p_prefix_size(p, [0, 1, 2], 1.0), Ok(1))
  assert_eq(min_p_prefix_size(p, [0, 1, 2], 0.0), Ok(3))
}

///|
test "epsilon is an absolute probability threshold" {
  let p = [0.5, 0.3, 0.2]
  assert_eq(epsilon_prefix_size(p, [0, 1, 2], 0.25), Ok(2))
  assert_eq(epsilon_prefix_size(p, [0, 1, 2], 0.75), Ok(1))
  assert_eq(sample_epsilon([3.0, 2.0, 1.0], 0.9, 0.9), Ok(0))
}

///|
test "min-p sampling respects surviving tokens" {
  assert_eq(sample_min_p([4.0, 0.0], 0.5, 0.99), Ok(0))
  assert_true(sample_min_p([1.0], 1.1, 0.5) is Err(_))
}
