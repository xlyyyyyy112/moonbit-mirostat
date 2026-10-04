///|
test "target sweep returns paired reports in target order" {
  let targets = [1.5, 2.5, 3.5]
  let points = sweep_targets(
    targets,
    0.1,
    Scenario::static_zipf(1.1),
    64,
    100,
    42,
  ).unwrap()
  assert_eq(points.length(), 3)
  for index in 0..<points.length() {
    assert_eq(points[index].target(), targets[index])
    assert_eq(points[index].comparison().v1().count(), 100)
    assert_eq(points[index].comparison().v2().count(), 100)
    assert_true(points[index].v1_error() >= 0.0)
    assert_true(points[index].v2_error() >= 0.0)
  }
}

///|
test "target sweep rejects invalid requests" {
  assert_true(sweep_targets([], 0.1, Scenario::uniform(), 8, 10, 1) is Err(_))
  assert_true(
    sweep_targets([0.0], 0.1, Scenario::uniform(), 8, 10, 1) is Err(_),
  )
}

///|
test "target sweep reveals an unreachable low target on this fixture" {
  let points = sweep_targets(
    [1.5, 2.5, 3.5],
    0.1,
    Scenario::static_zipf(1.15),
    256,
    1000,
    42,
  ).unwrap()
  assert_true(points[0].v1_error() > 0.5)
  assert_true(points[0].v2_error() > 0.5)
  assert_eq(calibrated_targets(points, 0.1), Ok([2.5, 3.5]))
  assert_true(calibrated_targets(points, -1.0) is Err(_))
}
