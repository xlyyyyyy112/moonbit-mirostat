///|
/// The current adaptive candidate distribution for an external categorical
/// sampler. Arrays align by index: `tokens[i]` has `conditional[i]` probability
/// after truncation and `original[i]` probability before truncation.
pub struct CandidateSet {
  tokens : Array[Int]
  conditional : Array[Double]
  original : Array[Double]
  mu : Double
} derive(Debug)

///|
pub fn CandidateSet::tokens(self : CandidateSet) -> Array[Int] {
  self.tokens.copy()
}

///|
pub fn CandidateSet::conditional(self : CandidateSet) -> Array[Double] {
  self.conditional.copy()
}

///|
pub fn CandidateSet::original(self : CandidateSet) -> Array[Double] {
  self.original.copy()
}

///|
pub fn CandidateSet::mu(self : CandidateSet) -> Double {
  self.mu
}

///|
/// Return the candidate-list position for a vocabulary token. This position
/// indexes the arrays returned by `conditional()` and `original()`.
pub fn CandidateSet::position(self : CandidateSet, token : Int) -> Int? {
  for index in 0..<self.tokens.length() {
    if self.tokens[index] == token {
      return Some(index)
    }
  }
  None
}

///|
/// Conditional probability after Mirostat truncation, or None if this token
/// is not in the current candidate set.
pub fn CandidateSet::conditional_probability(
  self : CandidateSet,
  token : Int,
) -> Double? {
  match self.position(token) {
    Some(index) => Some(self.conditional[index])
    None => None
  }
}

///|
/// Probability under the original model distribution, or None if this token
/// is not in the current candidate set.
pub fn CandidateSet::original_probability(
  self : CandidateSet,
  token : Int,
) -> Double? {
  match self.position(token) {
    Some(index) => Some(self.original[index])
    None => None
  }
}

///|
pub fn Sampler::candidates(
  self : Sampler,
  logits : Array[Double],
) -> Result[CandidateSet, SamplingError] {
  let p = match probabilities(logits) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  self.candidates_distribution(p)
}

///|
pub fn Sampler::candidates_weights(
  self : Sampler,
  weights : Array[Double],
) -> Result[CandidateSet, SamplingError] {
  let p = match normalize_weights(weights) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  self.candidates_distribution(p)
}

///|
fn Sampler::candidates_distribution(
  self : Sampler,
  p : Array[Double],
) -> Result[CandidateSet, SamplingError] {
  let tokens = match self.select_candidates(p) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let original : Array[Double] = []
  let mut mass = 0.0
  for token in tokens {
    let value = p[token]
    original.push(value)
    mass = mass + value
  }
  if !finite(mass) || mass <= 0.0 {
    return Err(NumericalFailure)
  }
  let conditional = original.map(fn(value) { value / mass })
  Ok({ tokens, conditional, original, mu: self.mu })
}
