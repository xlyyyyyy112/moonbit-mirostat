///|
test "comparison uses equal lengths and deterministic seeds" {
  let config = @mirostat.Config::new(2.0).unwrap()
  let first = compare_scenario(config, Scenario::bimodal(), 32, 100, 7).unwrap()
  let second = compare_scenario(config, Scenario::bimodal(), 32, 100, 7).unwrap()
  assert_eq(first, second)
  assert_eq(first.v1().count(), 100)
  assert_eq(first.v2().count(), 100)
  assert_true(first.v1().mean_surprise() >= 0.0)
  assert_true(first.v2().mean_surprise() >= 0.0)
}

///|
test "comparison rejects empty run and bad seed" {
  let config = @mirostat.Config::new(2.0).unwrap()
  assert_true(compare_scenario(config, Scenario::uniform(), 4, 0, 1) is Err(_))
  assert_true(compare_scenario(config, Scenario::uniform(), 4, 10, 0) is Err(_))
}
