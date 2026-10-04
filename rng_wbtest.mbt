///|
test "Park Miller has known first states" {
  let rng = ParkMiller::new(1).unwrap()
  assert_eq(rng.next_int(), 16807)
  assert_eq(rng.next_int(), 282475249)
}

///|
test "Park Miller draws fit sampler range" {
  let rng = ParkMiller::new(42).unwrap()
  for _ in 0..<100 {
    let draw = rng.next_uniform()
    assert_true(draw > 0.0 && draw < 1.0)
  }
  let sampler = Sampler::new(Config::new(2.0).unwrap())
  assert_true(sampler.sample_with_rng([2.0, 1.0], rng) is Ok(_))
}

///|
test "Park Miller rejects invalid seeds" {
  assert_true(ParkMiller::new(0) is Err(_))
  assert_true(ParkMiller::new(2147483647) is Err(_))
}
