///|
/// Online trace statistics for applications that do not want to retain every
/// step. Push only successful Mirostat decisions into this accumulator.
pub struct TraceAccumulator {
  target : Double
  mut count : Int
  mut total_surprise : Double
  mut total_kept : Double
  mut absolute_error : Double
  mut final_mu : Double
} derive(Debug)

///|
pub fn TraceAccumulator::new(
  target : Double,
) -> Result[TraceAccumulator, SamplingError] {
  if !finite(target) || target <= 0.0 {
    return Err(InvalidParameter("target must be finite and positive"))
  }
  Ok({
    target,
    count: 0,
    total_surprise: 0.0,
    total_kept: 0.0,
    absolute_error: 0.0,
    final_mu: 0.0,
  })
}

///|
pub fn TraceAccumulator::count(self : TraceAccumulator) -> Int {
  self.count
}

///|
pub fn TraceAccumulator::push(
  self : TraceAccumulator,
  step : Step,
) -> Result[Unit, SamplingError] {
  let difference = step.observed_surprise - self.target
  let next_surprise = self.total_surprise + step.observed_surprise
  let next_kept = self.total_kept + step.kept_tokens.to_double()
  let next_absolute = self.absolute_error +
    (if difference < 0.0 { -difference } else { difference })
  if !finite(next_surprise) ||
    !finite(next_kept) ||
    !finite(next_absolute) ||
    self.count == 2147483647 {
    return Err(NumericalFailure)
  }
  self.count = self.count + 1
  self.total_surprise = next_surprise
  self.total_kept = next_kept
  self.absolute_error = next_absolute
  self.final_mu = step.mu_after
  Ok(())
}

///|
pub fn TraceAccumulator::summary(
  self : TraceAccumulator,
) -> Result[TraceSummary, SamplingError] {
  if self.count == 0 {
    return Err(InvalidParameter("summary needs at least one step"))
  }
  let count = self.count.to_double()
  Ok({
    count: self.count,
    mean_surprise: self.total_surprise / count,
    mean_kept: self.total_kept / count,
    mean_absolute_error: self.absolute_error / count,
    final_mu: self.final_mu,
  })
}
