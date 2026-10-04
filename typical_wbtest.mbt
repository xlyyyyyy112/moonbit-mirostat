///|
test "typical order can differ from highest probability order" {
  let p = [0.7, 0.2, 0.1]
  let order = typical_order(p).unwrap()
  assert_eq(order.length(), 3)
  assert_true(order[0] != order[1])
  assert_eq(typical_prefix_size(p, order, 1.0), Ok(3))
}

///|
test "typical sampling rejects invalid thresholds" {
  assert_true(sample_typical([1.0, 0.0], 0.0, 0.5) is Err(_))
  assert_true(sample_typical([1.0, 0.0], 1.1, 0.5) is Err(_))
}

///|
test "typical order handles masked tail" {
  let p = [0.5, 0.5, 0.0]
  let order = typical_order(p).unwrap()
  assert_eq(order[2], 2)
  assert_true(sample_typical([0.0, 0.0, -1.0 / 0.0], 0.5, 0.3) is Ok(_))
}
