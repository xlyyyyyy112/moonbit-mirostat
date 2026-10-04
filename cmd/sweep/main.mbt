///|
/// Target-control fixture: each target starts a fresh sampler and uses the
/// same synthetic model and random seed.
fn main {
  let points = @experiments.sweep_targets(
    [1.5, 2.5, 3.5, 4.5],
    0.1,
    @experiments.Scenario::static_zipf(1.15),
    256,
    1000,
    42,
  ).unwrap()
  println(@experiments.sweep_csv(points))
}
