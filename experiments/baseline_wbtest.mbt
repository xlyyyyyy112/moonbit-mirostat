///|
test "baseline top one always selects first Zipf token" {
  let trace = run_baseline_static(
    Baseline::top_k(1),
    @mirostat.ParkMiller::new(3).unwrap(),
    zipf_logits(16, 1.0),
    20,
  ).unwrap()
  assert_eq(trace.length(), 20)
  for step in trace {
    assert_eq(step.token(), 0)
  }
  assert_true(baseline_mean_surprise(trace) is Ok(_))
}

///|
test "all baseline policies produce valid tokens" {
  let policies = [
    Baseline::top_k(2),
    Baseline::top_p(0.8),
    Baseline::temperature(1.0),
    Baseline::min_p(0.1),
    Baseline::epsilon(0.05),
    Baseline::typical(0.8),
  ]
  let row = zipf_logits(8, 1.1)
  for policy in policies {
    let token = policy.sample(row, 0.4).unwrap()
    assert_true(token >= 0 && token < 8)
  }
}

///|
test "fixed policy can be measured on a drifting model" {
  let trace = run_baseline_scenario(
    Baseline::top_k(4),
    @mirostat.ParkMiller::new(42).unwrap(),
    Scenario::drifting_zipf(0.7, 1.7),
    32,
    20,
  ).unwrap()
  assert_eq(trace.length(), 20)
  assert_true(baseline_mean_surprise(trace).unwrap() >= 0.0)
}
