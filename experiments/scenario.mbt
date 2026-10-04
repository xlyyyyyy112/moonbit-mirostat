///|
/// Deterministic synthetic model families for testing adaptation as the
/// distribution changes. They use ranked token ids, without text semantics.
pub enum Scenario {
  StaticZipf(Double)
  AlternatingZipf(Double, Double)
  DriftingZipf(Double, Double)
  Uniform
  Bimodal
  SpikeAndTail
} derive(Eq, Debug)

///|
pub fn Scenario::static_zipf(exponent : Double) -> Scenario {
  StaticZipf(exponent)
}

///|
pub fn Scenario::alternating_zipf(first : Double, second : Double) -> Scenario {
  AlternatingZipf(first, second)
}

///|
pub fn Scenario::drifting_zipf(start : Double, finish : Double) -> Scenario {
  DriftingZipf(start, finish)
}

///|
pub fn Scenario::uniform() -> Scenario {
  Uniform
}

///|
pub fn Scenario::bimodal() -> Scenario {
  Bimodal
}

///|
pub fn Scenario::spike_and_tail() -> Scenario {
  SpikeAndTail
}

///|
fn valid_exponent(value : Double) -> Bool {
  value == value && value > 0.0 && value != 1.0 / 0.0
}

///|
fn validate_scenario(
  scenario : Scenario,
) -> Result[Unit, @mirostat.SamplingError] {
  match scenario {
    StaticZipf(exponent) =>
      if !valid_exponent(exponent) {
        Err(@mirostat.SamplingError::invalid_parameter("invalid Zipf exponent"))
      } else {
        Ok(())
      }
    AlternatingZipf(first, second) | DriftingZipf(first, second) =>
      if !valid_exponent(first) || !valid_exponent(second) {
        Err(
          @mirostat.SamplingError::invalid_parameter("invalid Zipf exponents"),
        )
      } else {
        Ok(())
      }
    Uniform | Bimodal | SpikeAndTail => Ok(())
  }
}

///|
pub fn scenario_logits(
  scenario : Scenario,
  vocabulary : Int,
  index : Int,
  total_steps : Int,
) -> Result[Array[Double], @mirostat.SamplingError] {
  if vocabulary < 1 || index < 0 || total_steps < 1 || index >= total_steps {
    return Err(
      @mirostat.SamplingError::invalid_parameter("invalid scenario coordinates"),
    )
  }
  match validate_scenario(scenario) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  match scenario {
    StaticZipf(exponent) => Ok(zipf_logits(vocabulary, exponent))
    AlternatingZipf(first, second) =>
      Ok(zipf_logits(vocabulary, if index % 2 == 0 { first } else { second }))
    DriftingZipf(start, finish) => {
      let progress = if total_steps == 1 {
        0.0
      } else {
        index.to_double() / (total_steps - 1).to_double()
      }
      Ok(zipf_logits(vocabulary, start + (finish - start) * progress))
    }
    Uniform => Ok(Array::make(vocabulary, 0.0))
    Bimodal => {
      let row = Array::make(vocabulary, -2.0)
      row[0] = 3.0
      row[vocabulary / 2] = 3.0
      Ok(row)
    }
    SpikeAndTail => {
      let row = Array::make(vocabulary, 0.0)
      row[0] = 5.0
      Ok(row)
    }
  }
}

///|
pub fn run_scenario(
  sampler : @mirostat.Sampler,
  rng : @mirostat.ParkMiller,
  scenario : Scenario,
  vocabulary : Int,
  steps : Int,
) -> Result[Array[@mirostat.Step], @mirostat.SamplingError] {
  if steps < 0 || vocabulary < 1 {
    return Err(
      @mirostat.SamplingError::invalid_parameter("invalid scenario run"),
    )
  }
  match validate_scenario(scenario) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  let trace : Array[@mirostat.Step] = []
  for index in 0..<steps {
    let row = match scenario_logits(scenario, vocabulary, index, steps) {
      Ok(value) => value
      Err(error) => return Err(error)
    }
    match sampler.sample_with_rng(row, rng) {
      Ok(step) => trace.push(step)
      Err(error) => return Err(error)
    }
  }
  Ok(trace)
}
