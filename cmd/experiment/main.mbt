///|
/// A reproducible comparison on the same static model and random stream.
fn main {
  let config = @mirostat.Config::new(3.0).unwrap()
  let v1 = @mirostat.Sampler::new(config)
  let v2 = @mirostat.Sampler::new(config, version=@mirostat.Version::v2())
  let first = @experiments.run_static(
    v1,
    @mirostat.ParkMiller::new(42).unwrap(),
    256,
    1.15,
    1000,
  ).unwrap()
  let second = @experiments.run_static(
    v2,
    @mirostat.ParkMiller::new(42).unwrap(),
    256,
    1.15,
    1000,
  ).unwrap()
  let a = @mirostat.summarize(first, config.tau()).unwrap()
  let b = @mirostat.summarize(second, config.tau()).unwrap()
  println("variant,steps,mean_surprise,mean_absolute_error,mean_kept,final_mu")
  print_row("v1", a)
  print_row("v2", b)
  let row = @experiments.zipf_logits(256, 1.15)
  let fixed_k = @experiments.run_baseline_static(
    @experiments.Baseline::top_k(6),
    @mirostat.ParkMiller::new(42).unwrap(),
    row,
    1000,
  ).unwrap()
  let fixed_p = @experiments.run_baseline_static(
    @experiments.Baseline::top_p(0.9),
    @mirostat.ParkMiller::new(42).unwrap(),
    row,
    1000,
  ).unwrap()
  println(
    "top-k-6,1000," +
    @experiments.baseline_mean_surprise(fixed_k).unwrap().to_string() +
    ",,,",
  )
  println(
    "top-p-0.9,1000," +
    @experiments.baseline_mean_surprise(fixed_p).unwrap().to_string() +
    ",,,",
  )
  let drift_v1 = @experiments.run_scenario(
    @mirostat.Sampler::new(config),
    @mirostat.ParkMiller::new(42).unwrap(),
    @experiments.Scenario::drifting_zipf(0.7, 1.7),
    256,
    1000,
  ).unwrap()
  let drift_v2 = @experiments.run_scenario(
    @mirostat.Sampler::new(config, version=@mirostat.Version::v2()),
    @mirostat.ParkMiller::new(42).unwrap(),
    @experiments.Scenario::drifting_zipf(0.7, 1.7),
    256,
    1000,
  ).unwrap()
  print_row("drift-v1", @mirostat.summarize(drift_v1, config.tau()).unwrap())
  print_row("drift-v2", @mirostat.summarize(drift_v2, config.tau()).unwrap())
}

///|
fn print_row(name : String, report : @mirostat.TraceSummary) -> Unit {
  println(
    name +
    "," +
    report.count().to_string() +
    "," +
    report.mean_surprise().to_string() +
    "," +
    report.mean_absolute_error().to_string() +
    "," +
    report.mean_kept().to_string() +
    "," +
    report.final_mu().to_string(),
  )
}
