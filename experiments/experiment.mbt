///|
/// A deterministic model row with exact Zipf-shaped probabilities.
pub fn zipf_logits(vocabulary : Int, exponent : Double) -> Array[Double] {
  let row : Array[Double] = []
  if vocabulary <= 0 || exponent <= 0.0 {
    return row
  }
  for index in 0..<vocabulary {
    row.push(-exponent * @math.ln((index + 1).to_double()))
  }
  row
}

///|
/// Repeatedly sample a fixed Zipf-shaped model row. It is a diagnostic
/// fixture, not a substitute for a real context-dependent language model.
pub fn run_static(
  sampler : @mirostat.Sampler,
  rng : @mirostat.ParkMiller,
  vocabulary : Int,
  exponent : Double,
  steps : Int,
) -> Result[Array[@mirostat.Step], @mirostat.SamplingError] {
  if steps < 0 || vocabulary <= 0 || exponent <= 0.0 {
    return Err(
      @mirostat.SamplingError::invalid_parameter("invalid static experiment"),
    )
  }
  let logits = zipf_logits(vocabulary, exponent)
  let prepared = match @mirostat.PreparedRow::from_logits(logits) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let trace : Array[@mirostat.Step] = []
  for _ in 0..<steps {
    match sampler.sample_prepared(prepared, rng.next_uniform()) {
      Ok(step) => trace.push(step)
      Err(error) => return Err(error)
    }
  }
  Ok(trace)
}

///|
/// Alternate a peaked and a flat row to expose adaptation to changing
/// distributions while preserving a reproducible random stream.
pub fn run_alternating(
  sampler : @mirostat.Sampler,
  rng : @mirostat.ParkMiller,
  vocabulary : Int,
  peaked_exponent : Double,
  flat_exponent : Double,
  steps : Int,
) -> Result[Array[@mirostat.Step], @mirostat.SamplingError] {
  if steps < 0 ||
    vocabulary <= 0 ||
    peaked_exponent <= 0.0 ||
    flat_exponent <= 0.0 {
    return Err(
      @mirostat.SamplingError::invalid_parameter(
        "invalid alternating experiment",
      ),
    )
  }
  let peaked = zipf_logits(vocabulary, peaked_exponent)
  let flat = zipf_logits(vocabulary, flat_exponent)
  let peaked = match @mirostat.PreparedRow::from_logits(peaked) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let flat = match @mirostat.PreparedRow::from_logits(flat) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let trace : Array[@mirostat.Step] = []
  for index in 0..<steps {
    let row = if index % 2 == 0 { peaked } else { flat }
    match sampler.sample_prepared(row, rng.next_uniform()) {
      Ok(step) => trace.push(step)
      Err(error) => return Err(error)
    }
  }
  Ok(trace)
}
