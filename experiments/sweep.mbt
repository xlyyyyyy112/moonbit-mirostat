///|
/// One target value and both variants' observed summaries on a shared
/// synthetic model scenario.
pub struct SweepPoint {
  target : Double
  comparison : Comparison
} derive(Eq, Debug)

///|
pub fn SweepPoint::target(self : SweepPoint) -> Double {
  self.target
}

///|
pub fn SweepPoint::comparison(self : SweepPoint) -> Comparison {
  self.comparison
}

///|
pub fn SweepPoint::v1_error(self : SweepPoint) -> Double {
  let difference = self.comparison.v1().mean_surprise() - self.target
  if difference < 0.0 {
    -difference
  } else {
    difference
  }
}

///|
pub fn SweepPoint::v2_error(self : SweepPoint) -> Double {
  let difference = self.comparison.v2().mean_surprise() - self.target
  if difference < 0.0 {
    -difference
  } else {
    difference
  }
}

///|
/// Re-run a scenario for each target with the same seed and learning rate.
/// This checks controllability across targets rather than fitting one point.
pub fn sweep_targets(
  targets : Array[Double],
  eta : Double,
  scenario : Scenario,
  vocabulary : Int,
  steps : Int,
  seed : Int,
) -> Result[Array[SweepPoint], @mirostat.SamplingError] {
  if targets.length() == 0 {
    return Err(
      @mirostat.SamplingError::invalid_parameter(
        "target sweep must not be empty",
      ),
    )
  }
  let points : Array[SweepPoint] = []
  for target in targets {
    let config = match @mirostat.Config::new(target, eta~) {
      Ok(value) => value
      Err(error) => return Err(error)
    }
    let comparison = match
      compare_scenario(config, scenario, vocabulary, steps, seed) {
      Ok(value) => value
      Err(error) => return Err(error)
    }
    points.push({ target, comparison })
  }
  Ok(points)
}

///|
/// Retain target values where both variants' observed mean surprise falls
/// within the chosen tolerance. This describes only the tested scenario and
/// finite trace; it cannot certify reachability on another model.
pub fn calibrated_targets(
  points : Array[SweepPoint],
  tolerance : Double,
) -> Result[Array[Double], @mirostat.SamplingError] {
  if tolerance < 0.0 || tolerance != tolerance || tolerance == 1.0 / 0.0 {
    return Err(
      @mirostat.SamplingError::invalid_parameter(
        "tolerance must be finite and non-negative",
      ),
    )
  }
  let targets : Array[Double] = []
  for point in points {
    if point.v1_error() <= tolerance && point.v2_error() <= tolerance {
      targets.push(point.target())
    }
  }
  Ok(targets)
}
