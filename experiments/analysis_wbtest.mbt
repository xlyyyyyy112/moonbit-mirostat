///|
test "segment summaries preserve token count" {
  let sampler = @mirostat.Sampler::new(@mirostat.Config::new(2.0).unwrap())
  let trace = run_static(
    sampler,
    @mirostat.ParkMiller::new(9).unwrap(),
    16,
    1.0,
    11,
  ).unwrap()
  let segments = segment_summaries(trace, 2.0, 4).unwrap()
  assert_eq(segments.length(), 3)
  assert_eq(segments[0].count(), 4)
  assert_eq(segments[2].count(), 3)
  let errors = segment_mean_errors(trace, 2.0, 4).unwrap()
  assert_eq(errors.length(), 3)
}

///|
test "hit rate is bounded and validates input" {
  let sampler = @mirostat.Sampler::new(@mirostat.Config::new(2.0).unwrap())
  let trace = run_static(
    sampler,
    @mirostat.ParkMiller::new(9).unwrap(),
    16,
    1.0,
    10,
  ).unwrap()
  let rate = target_hit_rate(trace, 2.0, 0.5).unwrap()
  assert_true(rate >= 0.0 && rate <= 1.0)
  assert_true(target_hit_rate([], 2.0, 0.5) is Err(_))
  assert_true(segment_summaries(trace, 2.0, 0) is Err(_))
}
