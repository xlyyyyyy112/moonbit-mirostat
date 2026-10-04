///|
/// Dispersion diagnostics for token-level observed surprise. Variance is the
/// population variance in squared bits. Percentiles use nearest lower rank.
pub struct TraceSpread {
  minimum : Double
  maximum : Double
  variance : Double
  median : Double
  p90 : Double
  minimum_kept : Int
  maximum_kept : Int
} derive(Eq, Debug)

///|
pub fn TraceSpread::minimum(self : TraceSpread) -> Double {
  self.minimum
}

///|
pub fn TraceSpread::maximum(self : TraceSpread) -> Double {
  self.maximum
}

///|
pub fn TraceSpread::variance(self : TraceSpread) -> Double {
  self.variance
}

///|
pub fn TraceSpread::median(self : TraceSpread) -> Double {
  self.median
}

///|
pub fn TraceSpread::p90(self : TraceSpread) -> Double {
  self.p90
}

///|
pub fn TraceSpread::minimum_kept(self : TraceSpread) -> Int {
  self.minimum_kept
}

///|
pub fn TraceSpread::maximum_kept(self : TraceSpread) -> Int {
  self.maximum_kept
}

///|
pub fn trace_spread(trace : Array[Step]) -> Result[TraceSpread, SamplingError] {
  if trace.length() == 0 {
    return Err(InvalidParameter("spread needs at least one step"))
  }
  let values : Array[Double] = []
  let mut minimum = trace[0].observed_surprise
  let mut maximum = minimum
  let mut minimum_kept = trace[0].kept_tokens
  let mut maximum_kept = minimum_kept
  let mut mean = 0.0
  let mut moment = 0.0
  for index in 0..<trace.length() {
    let value = trace[index].observed_surprise
    if !finite(value) || value < 0.0 {
      return Err(NumericalFailure)
    }
    values.push(value)
    if value < minimum {
      minimum = value
    }
    if value > maximum {
      maximum = value
    }
    let kept = trace[index].kept_tokens
    if kept < minimum_kept {
      minimum_kept = kept
    }
    if kept > maximum_kept {
      maximum_kept = kept
    }
    let count = (index + 1).to_double()
    let delta = value - mean
    mean = mean + delta / count
    moment = moment + delta * (value - mean)
  }
  values.sort_by(fn(left, right) {
    if left < right {
      -1
    } else if left > right {
      1
    } else {
      0
    }
  })
  let last = values.length() - 1
  let median = values[(last.to_double() * 0.5).to_int()]
  let p90 = values[(last.to_double() * 0.9).to_int()]
  let variance = moment / values.length().to_double()
  if !finite(variance) {
    return Err(NumericalFailure)
  }
  Ok({
    minimum,
    maximum,
    variance: if variance < 0.0 {
      0.0
    } else {
      variance
    },
    median,
    p90,
    minimum_kept,
    maximum_kept,
  })
}
