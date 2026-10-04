///|
/// Aggregate diagnostics for a completed sequence of Mirostat steps.
pub struct TraceSummary {
  count : Int
  mean_surprise : Double
  mean_kept : Double
  mean_absolute_error : Double
  final_mu : Double
} derive(Eq, Debug)

///|
pub fn TraceSummary::count(self : TraceSummary) -> Int {
  self.count
}

///|
pub fn TraceSummary::mean_surprise(self : TraceSummary) -> Double {
  self.mean_surprise
}

///|
pub fn TraceSummary::mean_kept(self : TraceSummary) -> Double {
  self.mean_kept
}

///|
pub fn TraceSummary::mean_absolute_error(self : TraceSummary) -> Double {
  self.mean_absolute_error
}

///|
pub fn TraceSummary::final_mu(self : TraceSummary) -> Double {
  self.final_mu
}

///|
/// Summarize a trace against the requested surprise target.
pub fn summarize(
  steps : Array[Step],
  target : Double,
) -> Result[TraceSummary, SamplingError] {
  if steps.length() == 0 || !finite(target) || target <= 0.0 {
    return Err(InvalidParameter("summary needs steps and a positive target"))
  }
  let accumulator = match TraceAccumulator::new(target) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  for step in steps {
    match accumulator.push(step) {
      Ok(_) => ()
      Err(error) => return Err(error)
    }
  }
  accumulator.summary()
}

///|
/// Replay precomputed model rows and RNG draws, returning every decision.
/// A failed row leaves the sampler at the last successful step.
pub fn replay(
  sampler : Sampler,
  rows : Array[Array[Double]],
  draws : Array[Double],
) -> Result[Array[Step], SamplingError] {
  if rows.length() != draws.length() {
    return Err(InvalidParameter("one uniform draw is required per logits row"))
  }
  let output : Array[Step] = []
  for i in 0..<rows.length() {
    match sampler.sample(rows[i], draws[i]) {
      Ok(step) => output.push(step)
      Err(error) => return Err(error)
    }
  }
  Ok(output)
}

///|
/// Replay a batch as one transaction. If any row fails, feedback state is
/// restored to its value before the batch.
pub fn replay_atomic(
  sampler : Sampler,
  rows : Array[Array[Double]],
  draws : Array[Double],
) -> Result[Array[Step], SamplingError] {
  let before = sampler.checkpoint()
  match replay(sampler, rows, draws) {
    Ok(steps) => Ok(steps)
    Err(error) => {
      let _ = sampler.restore(before)
      Err(error)
    }
  }
}

///|
/// Cumulative average surprise after each generated token. This is the
/// observed cross-entropy estimate used to assess long-run target control.
pub fn running_surprise(steps : Array[Step]) -> Array[Double] {
  let means : Array[Double] = []
  let mut total = 0.0
  for index in 0..<steps.length() {
    total = total + steps[index].observed_surprise
    means.push(total / (index + 1).to_double())
  }
  means
}

///|
/// Average surprise over the most recent `window` tokens at every position.
/// The initial entries use all observations available so far.
pub fn rolling_surprise(
  steps : Array[Step],
  window : Int,
) -> Result[Array[Double], SamplingError] {
  if window < 1 {
    return Err(InvalidParameter("rolling window must be positive"))
  }
  let values : Array[Double] = []
  let mut total = 0.0
  for index in 0..<steps.length() {
    total = total + steps[index].observed_surprise
    if index >= window {
      total = total - steps[index - window].observed_surprise
    }
    let count = if index + 1 < window { index + 1 } else { window }
    values.push(total / count.to_double())
  }
  Ok(values)
}
