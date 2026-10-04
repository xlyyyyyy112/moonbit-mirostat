///|
test "softmax remains finite when logits share a large offset" {
  let result = probabilities([1000.0, 1000.0])
  assert_true(result is Ok(_))
  match result {
    Ok(p) => {
      assert_eq(p.length(), 2)
      assert_true(p[0] > 0.499999 && p[0] < 0.500001)
      assert_true(p[1] > 0.499999 && p[1] < 0.500001)
    }
    Err(_) => fail("softmax failed")
  }
}

///|
test "ranking breaks ties by token id" {
  assert_eq(rank([0.2, 0.4, 0.4]), Ok([1, 2, 0]))
}

///|
test "draw respects prefix and uniform boundary" {
  let p = [0.6, 0.3, 0.1]
  let order = [0, 1, 2]
  assert_eq(draw_prefix(p, order, 2, 0.0), Ok(0))
  assert_eq(draw_prefix(p, order, 2, 0.999), Ok(1))
  assert_eq(draw_prefix(p, order, 2, 1.0), Err(InvalidUniform))
}

///|
test "surprise of fair coin is one bit" {
  match surprise(0.5) {
    Ok(value) => assert_true(value > 0.999999 && value < 1.000001)
    Err(_) => fail("surprise failed")
  }
}

///|
test "bad logits and weights are refused" {
  assert_eq(probabilities([]), Err(EmptyLogits))
  assert_eq(probabilities([0.0, 1.0 / 0.0]), Err(NonFiniteLogit(1)))
  assert_eq(rank([0.5, -0.5]), Err(InvalidProbability(1)))
}

///|
test "negative infinity masks a token" {
  let masked = -1.0 / 0.0
  assert_eq(probabilities([0.0, masked]).unwrap(), [1.0, 0.0])
  assert_eq(probabilities([masked, masked]), Err(AllMasked))
  assert_eq(sample_top_k([masked, 0.0], 1, 0.5), Ok(1))
}

///|
test "malformed permutation cannot trap public draw" {
  assert_true(draw_prefix([0.5, 0.5], [0, 2], 1, 0.5) is Err(_))
  assert_true(draw_prefix([0.5, 0.5], [0, 0], 1, 0.5) is Err(_))
}

///|
test "weight normalization tolerates a common huge scale" {
  let p = normalize_weights([1.0e300, 1.0e300]).unwrap()
  assert_eq(p, [0.5, 0.5])
  assert_true(normalize_weights([0.0, 0.0]) is Err(_))
}

///|
test "sampling errors have concise messages" {
  assert_eq(EmptyLogits.message(), "logits or weights must not be empty")
  assert_eq(
    NonFiniteLogit(3).message(),
    "logit at token 3 is NaN or positive infinity",
  )
}
