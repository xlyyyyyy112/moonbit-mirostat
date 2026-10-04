///|
/// Fixed sampling policies for experiments against adaptive Mirostat.
pub enum Baseline {
  TopK(Int)
  TopP(Double)
  Temperature(Double)
  MinP(Double)
  Epsilon(Double)
  Typical(Double)
} derive(Eq, Debug)

///|
pub fn Baseline::top_k(k : Int) -> Baseline {
  TopK(k)
}

///|
pub fn Baseline::top_p(p : Double) -> Baseline {
  TopP(p)
}

///|
pub fn Baseline::temperature(t : Double) -> Baseline {
  Temperature(t)
}

///|
pub fn Baseline::min_p(p : Double) -> Baseline {
  MinP(p)
}

///|
pub fn Baseline::epsilon(p : Double) -> Baseline {
  Epsilon(p)
}

///|
pub fn Baseline::typical(p : Double) -> Baseline {
  Typical(p)
}

///|
pub struct BaselineStep {
  token : Int
  original_probability : Double
  observed_surprise : Double
} derive(Eq, Debug)

///|
pub fn BaselineStep::token(self : BaselineStep) -> Int {
  self.token
}

///|
pub fn BaselineStep::original_probability(self : BaselineStep) -> Double {
  self.original_probability
}

///|
pub fn BaselineStep::observed_surprise(self : BaselineStep) -> Double {
  self.observed_surprise
}

///|
pub fn Baseline::sample(
  self : Baseline,
  logits : Array[Double],
  uniform : Double,
) -> Result[Int, @mirostat.SamplingError] {
  match self {
    TopK(k) => @mirostat.sample_top_k(logits, k, uniform)
    TopP(p) => @mirostat.sample_top_p(logits, p, uniform)
    Temperature(t) => @mirostat.sample_temperature(logits, t, uniform)
    MinP(p) => @mirostat.sample_min_p(logits, p, uniform)
    Epsilon(p) => @mirostat.sample_epsilon(logits, p, uniform)
    Typical(p) => @mirostat.sample_typical(logits, p, uniform)
  }
}

///|
/// Replay a fixed policy on a single model row with the same RNG interface
/// used by the adaptive experiment. Surprise uses original model probability.
pub fn run_baseline_static(
  policy : Baseline,
  rng : @mirostat.ParkMiller,
  logits : Array[Double],
  steps : Int,
) -> Result[Array[BaselineStep], @mirostat.SamplingError] {
  if steps < 0 {
    return Err(
      @mirostat.SamplingError::invalid_parameter("steps must be non-negative"),
    )
  }
  let probabilities = match @mirostat.probabilities(logits) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let trace : Array[BaselineStep] = []
  for _ in 0..<steps {
    let token = match policy.sample(logits, rng.next_uniform()) {
      Ok(value) => value
      Err(error) => return Err(error)
    }
    let observed = match @mirostat.surprise(probabilities[token]) {
      Ok(value) => value
      Err(error) => return Err(error)
    }
    trace.push({
      token,
      original_probability: probabilities[token],
      observed_surprise: observed,
    })
  }
  Ok(trace)
}

///|
pub fn baseline_mean_surprise(
  trace : Array[BaselineStep],
) -> Result[Double, @mirostat.SamplingError] {
  if trace.length() == 0 {
    return Err(
      @mirostat.SamplingError::invalid_parameter("trace must not be empty"),
    )
  }
  let mut total = 0.0
  for step in trace {
    total = total + step.observed_surprise
  }
  Ok(total / trace.length().to_double())
}

///|
/// Replay a fixed policy as a synthetic model scenario changes over time.
/// Each step's surprise uses that step's original, untruncated distribution.
pub fn run_baseline_scenario(
  policy : Baseline,
  rng : @mirostat.ParkMiller,
  scenario : Scenario,
  vocabulary : Int,
  steps : Int,
) -> Result[Array[BaselineStep], @mirostat.SamplingError] {
  if vocabulary < 1 || steps < 0 {
    return Err(
      @mirostat.SamplingError::invalid_parameter("invalid baseline scenario"),
    )
  }
  let trace : Array[BaselineStep] = []
  for index in 0..<steps {
    let row = match scenario_logits(scenario, vocabulary, index, steps) {
      Ok(value) => value
      Err(error) => return Err(error)
    }
    let p = match @mirostat.probabilities(row) {
      Ok(value) => value
      Err(error) => return Err(error)
    }
    let token = match policy.sample(row, rng.next_uniform()) {
      Ok(value) => value
      Err(error) => return Err(error)
    }
    let observed = match @mirostat.surprise(p[token]) {
      Ok(value) => value
      Err(error) => return Err(error)
    }
    trace.push({
      token,
      original_probability: p[token],
      observed_surprise: observed,
    })
  }
  Ok(trace)
}
