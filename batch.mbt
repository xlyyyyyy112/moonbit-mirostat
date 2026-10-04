///|
/// Sample one next token for each independent generation session. Input arrays
/// must align by index. On an error, earlier sessions have already advanced.
pub fn sample_batch(
  samplers : Array[Sampler],
  logits : Array[Array[Double]],
  uniforms : Array[Double],
) -> Result[Array[Step], SamplingError] {
  if samplers.length() != logits.length() ||
    samplers.length() != uniforms.length() {
    return Err(InvalidParameter("batch arrays must have the same length"))
  }
  let output : Array[Step] = []
  for i in 0..<samplers.length() {
    match samplers[i].sample(logits[i], uniforms[i]) {
      Ok(step) => output.push(step)
      Err(error) => return Err(error)
    }
  }
  Ok(output)
}

///|
/// Sample all sessions as a transaction. If any row fails, every sampler is
/// restored to its previous feedback state. External RNG state is unchanged
/// because uniforms are supplied as values.
pub fn sample_batch_atomic(
  samplers : Array[Sampler],
  logits : Array[Array[Double]],
  uniforms : Array[Double],
) -> Result[Array[Step], SamplingError] {
  if samplers.length() != logits.length() ||
    samplers.length() != uniforms.length() {
    return Err(InvalidParameter("batch arrays must have the same length"))
  }
  let checkpoints : Array[Checkpoint] = []
  for sampler in samplers {
    checkpoints.push(sampler.checkpoint())
  }
  match sample_batch(samplers, logits, uniforms) {
    Ok(steps) => Ok(steps)
    Err(error) => {
      for i in 0..<samplers.length() {
        let _ = samplers[i].restore(checkpoints[i])
      }
      Err(error)
    }
  }
}

///|
/// Draw one token for each independent session from a shared prepared model
/// row. This is useful for Monte Carlo comparisons using one fixed logits row
/// while each session maintains its own Mirostat feedback state.
pub fn sample_shared_prepared(
  samplers : Array[Sampler],
  row : PreparedRow,
  uniforms : Array[Double],
) -> Result[Array[Step], SamplingError] {
  if samplers.length() != uniforms.length() {
    return Err(InvalidParameter("one uniform draw is required per sampler"))
  }
  let output : Array[Step] = []
  for index in 0..<samplers.length() {
    match samplers[index].sample_prepared(row, uniforms[index]) {
      Ok(step) => output.push(step)
      Err(error) => return Err(error)
    }
  }
  Ok(output)
}
