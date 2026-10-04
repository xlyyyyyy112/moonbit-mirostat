///|
test "partial top indices agree with complete ranking" {
  let p = [0.1, 0.3, 0.2, 0.3, 0.0, 0.1]
  let full = rank(p).unwrap()
  for k in 1..<=p.length() {
    let partial = top_indices(p, k).unwrap()
    for i in 0..<k {
      assert_eq(partial[i], full[i])
    }
  }
}

///|
test "partial selection rejects invalid inputs" {
  assert_true(top_indices([], 1) is Err(_))
  assert_true(top_indices([0.5], 0) is Err(_))
  assert_true(top_indices([-1.0], 1) is Err(_))
}

///|
test "partial selection matches full rank on deterministic grid" {
  let p : Array[Double] = []
  for i in 0..<257 {
    p.push(((i * 37 + 11) % 53).to_double())
  }
  let full = rank(p).unwrap()
  for k in [1, 3, 17, 64, 256] {
    let partial = top_indices(p, k).unwrap()
    for i in 0..<k {
      assert_eq(partial[i], full[i])
    }
  }
}
