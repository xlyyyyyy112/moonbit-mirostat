///|
/// Advance Mirostat feedback after an external sampler chooses a token from
/// this library's candidate set. The token is checked against the same
/// adaptive truncation that `sample` would use. This lets a host own the RNG
/// and categorical draw while MoonBit maintains the feedback state.
pub fn Sampler::observe_token(
  self : Sampler,
  logits : Array[Double],
  token : Int,
) -> Result[Step, SamplingError] {
  let p = match probabilities(logits) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  self.observe_distribution(p, token)
}

///|
pub fn Sampler::observe_weighted_token(
  self : Sampler,
  weights : Array[Double],
  token : Int,
) -> Result[Step, SamplingError] {
  let p = match normalize_weights(weights) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  self.observe_distribution(p, token)
}

///|
fn Sampler::observe_distribution(
  self : Sampler,
  p : Array[Double],
  token : Int,
) -> Result[Step, SamplingError] {
  let candidates = match self.select_candidates(p) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  self.observe_candidates(p, candidates, token)
}

///|
/// Validate an externally chosen token against a precomputed model row.
pub fn Sampler::observe_prepared_token(
  self : Sampler,
  row : PreparedRow,
  token : Int,
) -> Result[Step, SamplingError] {
  let keep = match self.prepared_keep(row) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let candidates : Array[Int] = []
  for index in 0..<keep {
    candidates.push(row.order[index])
  }
  self.observe_candidates(row.probabilities, candidates, token)
}

///|
fn Sampler::observe_candidates(
  self : Sampler,
  p : Array[Double],
  candidates : Array[Int],
  token : Int,
) -> Result[Step, SamplingError] {
  if token < 0 || token >= p.length() {
    return Err(InvalidParameter("observed token outside vocabulary"))
  }
  let mut allowed = false
  for candidate in candidates {
    if candidate == token {
      allowed = true
      break
    }
  }
  if !allowed || p[token] <= 0.0 {
    return Err(
      InvalidParameter("observed token outside Mirostat candidate set"),
    )
  }
  self.commit_step(p, token, candidates.length())
}
