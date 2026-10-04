///|
/// Candidate-set diagnostics computed without drawing a token or updating mu.
pub struct Preview {
  vocabulary : Int
  kept_tokens : Int
  retained_mass : Double
  expected_surprise : Double
  mu : Double
} derive(Eq, Debug)

///|
pub fn Preview::vocabulary(self : Preview) -> Int {
  self.vocabulary
}

///|
pub fn Preview::kept_tokens(self : Preview) -> Int {
  self.kept_tokens
}

///|
pub fn Preview::retained_mass(self : Preview) -> Double {
  self.retained_mass
}

///|
pub fn Preview::expected_surprise(self : Preview) -> Double {
  self.expected_surprise
}

///|
pub fn Preview::mu(self : Preview) -> Double {
  self.mu
}

///|
/// Inspect the next candidate set on a logits row without consuming a draw.
pub fn Sampler::preview(
  self : Sampler,
  logits : Array[Double],
) -> Result[Preview, SamplingError] {
  let p = match probabilities(logits) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  self.preview_distribution(p)
}

///|
pub fn Sampler::preview_weights(
  self : Sampler,
  weights : Array[Double],
) -> Result[Preview, SamplingError] {
  let p = match normalize_weights(weights) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  self.preview_distribution(p)
}

///|
fn Sampler::preview_distribution(
  self : Sampler,
  p : Array[Double],
) -> Result[Preview, SamplingError] {
  let order = match self.select_candidates(p) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let mut mass = 0.0
  let mut weighted_surprise = 0.0
  for token in order {
    let value = p[token]
    mass = mass + value
    if value > 0.0 {
      weighted_surprise = weighted_surprise -
        value * @math.ln(value) / @math.ln(2.0)
    }
  }
  if !finite(mass) || mass <= 0.0 {
    return Err(NumericalFailure)
  }
  Ok({
    vocabulary: p.length(),
    kept_tokens: order.length(),
    retained_mass: mass,
    expected_surprise: weighted_surprise / mass,
    mu: self.mu,
  })
}
