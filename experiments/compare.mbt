///|
/// Reproducible side-by-side summary of both Mirostat variants on the same
/// synthetic scenario and Park-Miller random stream.
pub struct Comparison {
  v1 : @mirostat.TraceSummary
  v2 : @mirostat.TraceSummary
} derive(Eq, Debug)

///|
pub fn Comparison::v1(self : Comparison) -> @mirostat.TraceSummary {
  self.v1
}

///|
pub fn Comparison::v2(self : Comparison) -> @mirostat.TraceSummary {
  self.v2
}

///|
pub fn Comparison::mean_surprise_gap(self : Comparison) -> Double {
  self.v1.mean_surprise() - self.v2.mean_surprise()
}

///|
pub fn compare_scenario(
  config : @mirostat.Config,
  scenario : Scenario,
  vocabulary : Int,
  steps : Int,
  seed : Int,
) -> Result[Comparison, @mirostat.SamplingError] {
  if steps < 1 {
    return Err(
      @mirostat.SamplingError::invalid_parameter(
        "comparison needs at least one step",
      ),
    )
  }
  let first_rng = match @mirostat.ParkMiller::new(seed) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let second_rng = match @mirostat.ParkMiller::new(seed) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let first = match
    run_scenario(
      @mirostat.Sampler::new(config, version=@mirostat.Version::v1()),
      first_rng,
      scenario,
      vocabulary,
      steps,
    ) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let second = match
    run_scenario(
      @mirostat.Sampler::new(config, version=@mirostat.Version::v2()),
      second_rng,
      scenario,
      vocabulary,
      steps,
    ) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let v1 = match @mirostat.summarize(first, config.tau()) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let v2 = match @mirostat.summarize(second, config.tau()) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  Ok({ v1, v2, })
}
