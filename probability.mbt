///|
/// Errors are values so callers can decide whether to skip a malformed model row.
pub enum SamplingError {
  EmptyLogits
  AllMasked
  NonFiniteLogit(Int)
  InvalidUniform
  InvalidProbability(Int)
  InvalidParameter(String)
  NumericalFailure
} derive(Eq, Debug)

///|
pub fn SamplingError::invalid_parameter(message : String) -> SamplingError {
  InvalidParameter(message)
}

///|
pub fn SamplingError::message(self : SamplingError) -> String {
  match self {
    EmptyLogits => "logits or weights must not be empty"
    AllMasked => "at least one finite logit must remain unmasked"
    NonFiniteLogit(index) =>
      "logit at token " + index.to_string() + " is NaN or positive infinity"
    InvalidUniform => "uniform draw must be finite and in [0, 1)"
    InvalidProbability(index) =>
      "weight at token " + index.to_string() + " is invalid"
    InvalidParameter(message) => message
    NumericalFailure => "sampling calculation failed numerically"
  }
}

///|
fn finite(value : Double) -> Bool {
  value == value && value != 1.0 / 0.0 && value != -1.0 / 0.0
}

///|
fn check_logits(logits : Array[Double]) -> Result[Unit, SamplingError] {
  if logits.length() == 0 {
    return Err(EmptyLogits)
  }
  let mut usable = 0
  for i in 0..<logits.length() {
    if logits[i] != logits[i] || logits[i] == 1.0 / 0.0 {
      return Err(NonFiniteLogit(i))
    }
    if finite(logits[i]) {
      usable = usable + 1
    }
  }
  if usable == 0 {
    return Err(AllMasked)
  }
  Ok(())
}

///|
/// Stable softmax. Subtracting the maximum avoids overflow for large logits.
pub fn probabilities(
  logits : Array[Double],
) -> Result[Array[Double], SamplingError] {
  match check_logits(logits) {
    Err(e) => return Err(e)
    Ok(_) => ()
  }
  let mut maximum = logits[0]
  for value in logits {
    if value > maximum {
      maximum = value
    }
  }
  let weights : Array[Double] = []
  let mut total = 0.0
  for value in logits {
    let weight = @math.exp(value - maximum)
    weights.push(weight)
    total = total + weight
  }
  if !finite(total) || total <= 0.0 {
    return Err(NumericalFailure)
  }
  weights.map(fn(weight) { weight / total }) |> Ok
}

///|
/// Normalize externally supplied non-negative weights. This is useful when a
/// model backend already supplies probabilities rather than logits.
pub fn normalize_weights(
  weights : Array[Double],
) -> Result[Array[Double], SamplingError] {
  if weights.length() == 0 {
    return Err(EmptyLogits)
  }
  let mut maximum = 0.0
  for i in 0..<weights.length() {
    let value = weights[i]
    if !finite(value) || value < 0.0 {
      return Err(InvalidProbability(i))
    }
    if value > maximum {
      maximum = value
    }
  }
  if maximum == 0.0 {
    return Err(NumericalFailure)
  }
  let scaled : Array[Double] = []
  let mut total = 0.0
  for weight in weights {
    let value = weight / maximum
    scaled.push(value)
    total = total + value
  }
  if !finite(total) || total <= 0.0 {
    return Err(NumericalFailure)
  }
  Ok(scaled.map(fn(value) { value / total }))
}

///|
/// Sort token ids by descending probability; ties favor the lower id.
pub fn rank(probabilities : Array[Double]) -> Result[Array[Int], SamplingError] {
  if probabilities.length() == 0 {
    return Err(EmptyLogits)
  }
  let order : Array[Int] = []
  for i in 0..<probabilities.length() {
    let p = probabilities[i]
    if !finite(p) || p < 0.0 {
      return Err(InvalidProbability(i))
    }
    order.push(i)
  }
  order.sort_by(fn(left, right) {
    if probabilities[left] > probabilities[right] {
      -1
    } else if probabilities[left] < probabilities[right] {
      1
    } else {
      left.compare(right)
    }
  })
  Ok(order)
}

///|
fn check_order(
  probabilities : Array[Double],
  order : Array[Int],
) -> Result[Unit, SamplingError] {
  if probabilities.length() == 0 || probabilities.length() != order.length() {
    return Err(InvalidParameter("order must cover the vocabulary"))
  }
  let seen = Array::make(order.length(), false)
  for token in order {
    if token < 0 || token >= order.length() || seen[token] {
      return Err(InvalidParameter("order must be a token permutation"))
    }
    seen[token] = true
    if !finite(probabilities[token]) || probabilities[token] < 0.0 {
      return Err(InvalidProbability(token))
    }
  }
  Ok(())
}

///|
/// Draw from a ranked prefix using one externally supplied uniform number.
/// The caller controls randomness, making a run exactly replayable.
pub fn draw_prefix(
  probabilities : Array[Double],
  order : Array[Int],
  keep : Int,
  uniform : Double,
) -> Result[Int, SamplingError] {
  if !finite(uniform) || uniform < 0.0 || uniform >= 1.0 {
    return Err(InvalidUniform)
  }
  if keep <= 0 ||
    keep > order.length() ||
    probabilities.length() != order.length() {
    return Err(InvalidParameter("keep must be between one and vocabulary size"))
  }
  match check_order(probabilities, order) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  draw_valid_prefix(probabilities, order, keep, uniform)
}

///|
/// Draw from an arbitrary list of distinct candidate token indices. This
/// accepts a partial vocabulary and retains the supplied candidate order.
pub fn draw_candidates(
  probabilities : Array[Double],
  candidates : Array[Int],
  uniform : Double,
) -> Result[Int, SamplingError] {
  if !finite(uniform) || uniform < 0.0 || uniform >= 1.0 {
    return Err(InvalidUniform)
  }
  if probabilities.length() == 0 ||
    candidates.length() == 0 ||
    candidates.length() > probabilities.length() {
    return Err(InvalidParameter("candidate list must be nonempty"))
  }
  let seen = Array::make(probabilities.length(), false)
  for token in candidates {
    if token < 0 || token >= probabilities.length() || seen[token] {
      return Err(InvalidParameter("invalid or repeated candidate token"))
    }
    seen[token] = true
    if !finite(probabilities[token]) || probabilities[token] < 0.0 {
      return Err(InvalidProbability(token))
    }
  }
  draw_valid_prefix(probabilities, candidates, candidates.length(), uniform)
}

///|
fn draw_valid_prefix(
  probabilities : Array[Double],
  order : Array[Int],
  keep : Int,
  uniform : Double,
) -> Result[Int, SamplingError] {
  let mut total = 0.0
  for i in 0..<keep {
    total = total + probabilities[order[i]]
  }
  if !finite(total) || total <= 0.0 {
    return Err(NumericalFailure)
  }
  let threshold = uniform * total
  let mut cumulative = 0.0
  let mut last_positive = order[0]
  for i in 0..<keep {
    let token = order[i]
    let weight = probabilities[token]
    if weight > 0.0 {
      last_positive = token
    }
    cumulative = cumulative + weight
    if threshold < cumulative {
      return Ok(token)
    }
  }
  Ok(last_positive)
}

///|
/// Information content in bits under the untruncated model distribution.
pub fn surprise(probability : Double) -> Result[Double, SamplingError] {
  if !finite(probability) || probability <= 0.0 || probability > 1.0 {
    return Err(InvalidProbability(0))
  }
  Ok(-@math.ln(probability) / @math.ln(2.0))
}
