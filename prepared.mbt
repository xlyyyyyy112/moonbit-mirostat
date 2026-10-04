///|
/// Immutable-by-convention model row prepared for repeated draws. The
/// probabilities and complete rank order are computed only once.
pub struct PreparedRow {
  probabilities : Array[Double]
  order : Array[Int]
} derive(Debug)

///|
pub fn PreparedRow::from_logits(
  logits : Array[Double],
) -> Result[PreparedRow, SamplingError] {
  let p = match probabilities(logits) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  PreparedRow::from_normalized(p)
}

///|
pub fn PreparedRow::from_weights(
  weights : Array[Double],
) -> Result[PreparedRow, SamplingError] {
  let p = match normalize_weights(weights) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  PreparedRow::from_normalized(p)
}

///|
fn PreparedRow::from_normalized(
  p : Array[Double],
) -> Result[PreparedRow, SamplingError] {
  let order = match rank(p) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  Ok({ probabilities: p, order })
}

///|
pub fn PreparedRow::vocabulary(self : PreparedRow) -> Int {
  self.probabilities.length()
}

///|
pub fn PreparedRow::probability(
  self : PreparedRow,
  token : Int,
) -> Result[Double, SamplingError] {
  if token < 0 || token >= self.probabilities.length() {
    return Err(InvalidParameter("token outside prepared vocabulary"))
  }
  Ok(self.probabilities[token])
}

///|
/// Draw using a prepared model row. Useful in simulations and repeated
/// sampling from unchanged logits; real autoregressive models usually return
/// a different row each step and can use `sample` directly.
pub fn Sampler::sample_prepared(
  self : Sampler,
  row : PreparedRow,
  uniform : Double,
) -> Result[Step, SamplingError] {
  let keep = match self.prepared_keep(row) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  if !finite(uniform) || uniform < 0.0 || uniform >= 1.0 {
    return Err(InvalidUniform)
  }
  let token = match
    draw_valid_prefix(row.probabilities, row.order, keep, uniform) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  self.commit_step(row.probabilities, token, keep)
}

///|
fn Sampler::prepared_keep(
  self : Sampler,
  row : PreparedRow,
) -> Result[Int, SamplingError] {
  match self.version {
    V1 =>
      if row.vocabulary() == 1 {
        Ok(1)
      } else {
        let exponent = match
          zipf_exponent(row.probabilities, row.order, self.config.m) {
          Ok(value) => value
          Err(error) => return Err(error)
        }
        estimated_k(self.mu, exponent, row.vocabulary())
      }
    V2 => v2_prefix_size(row.probabilities, row.order, self.mu)
  }
}

///|
/// Preview a prepared row without redoing softmax or ranking.
pub fn Sampler::preview_prepared(
  self : Sampler,
  row : PreparedRow,
) -> Result[Preview, SamplingError] {
  let keep = match self.prepared_keep(row) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let mass = match prefix_mass(row.probabilities, row.order, keep) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let expected = match
    expected_prefix_surprise(row.probabilities, row.order, keep) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  Ok({
    vocabulary: row.vocabulary(),
    kept_tokens: keep,
    retained_mass: mass,
    expected_surprise: expected,
    mu: self.mu,
  })
}
