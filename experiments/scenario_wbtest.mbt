///|
test "drift moves continuously from first to last exponent" {
  let scenario = Scenario::drifting_zipf(0.5, 2.0)
  let first = scenario_logits(scenario, 4, 0, 3).unwrap()
  let middle = scenario_logits(scenario, 4, 1, 3).unwrap()
  let last = scenario_logits(scenario, 4, 2, 3).unwrap()
  assert_true(first[3] > middle[3])
  assert_true(middle[3] > last[3])
}

///|
test "uniform scenario is valid for both Mirostat versions" {
  let config = @mirostat.Config::new(2.0).unwrap()
  for version in [@mirostat.Version::v1(), @mirostat.Version::v2()] {
    let sampler = @mirostat.Sampler::new(config, version~)
    let trace = run_scenario(
      sampler,
      @mirostat.ParkMiller::new(5).unwrap(),
      Scenario::uniform(),
      8,
      10,
    ).unwrap()
    assert_eq(trace.length(), 10)
  }
}

///|
test "scenario rejects bad coordinates and exponents" {
  assert_true(scenario_logits(Scenario::uniform(), 0, 0, 1) is Err(_))
  assert_true(scenario_logits(Scenario::static_zipf(0.0), 4, 0, 1) is Err(_))
  assert_true(scenario_logits(Scenario::uniform(), 4, 1, 1) is Err(_))
}

///|
test "drifting Zipf run stays near surprise target" {
  let config = @mirostat.Config::new(3.0).unwrap()
  let trace = run_scenario(
    @mirostat.Sampler::new(config, version=@mirostat.Version::v2()),
    @mirostat.ParkMiller::new(42).unwrap(),
    Scenario::drifting_zipf(0.7, 1.7),
    256,
    1000,
  ).unwrap()
  let summary = @mirostat.summarize(trace, config.tau()).unwrap()
  assert_true(summary.mean_surprise() > 2.7)
  assert_true(summary.mean_surprise() < 3.3)
}

///|
test "non Zipf scenarios expose two distinct shapes" {
  let bimodal = scenario_logits(Scenario::bimodal(), 8, 0, 1).unwrap()
  let spike = scenario_logits(Scenario::spike_and_tail(), 8, 0, 1).unwrap()
  assert_eq(bimodal[0], bimodal[4])
  assert_true(bimodal[0] > bimodal[1])
  assert_true(spike[0] > spike[4])
  assert_eq(spike[1], spike[4])
}
