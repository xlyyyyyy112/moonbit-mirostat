///|
/// Apply an allow-mask. This is useful when a grammar or host application
/// restricts the next token vocabulary. At least one allowed finite logit is
/// required. The input array is not mutated.
pub fn mask_logits(
  logits : Array[Double],
  allowed : Array[Bool],
) -> Result[Array[Double], SamplingError] {
  if logits.length() != allowed.length() {
    return Err(InvalidParameter("mask length must match vocabulary"))
  }
  match check_logits(logits) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  let output : Array[Double] = []
  for i in 0..<logits.length() {
    output.push(if allowed[i] { logits[i] } else { -1.0 / 0.0 })
  }
  match check_logits(output) {
    Ok(_) => Ok(output)
    Err(error) => Err(error)
  }
}

///|
/// Mask a sparse list of forbidden token ids without building a full Boolean
/// mask in the caller. Repeated ids have the same effect as one occurrence.
pub fn block_token_ids(
  logits : Array[Double],
  blocked : Array[Int],
) -> Result[Array[Double], SamplingError] {
  let allowed = Array::make(logits.length(), true)
  for token in blocked {
    if token < 0 || token >= logits.length() {
      return Err(InvalidParameter("blocked token outside vocabulary"))
    }
    allowed[token] = false
  }
  mask_logits(logits, allowed)
}

///|
/// Keep only a sparse set of allowed token ids. A valid nonempty support must
/// remain after applying the allowlist and any masks already in the logits.
pub fn allow_token_ids(
  logits : Array[Double],
  allowed_tokens : Array[Int],
) -> Result[Array[Double], SamplingError] {
  let allowed = Array::make(logits.length(), false)
  for token in allowed_tokens {
    if token < 0 || token >= logits.length() {
      return Err(InvalidParameter("allowed token outside vocabulary"))
    }
    allowed[token] = true
  }
  mask_logits(logits, allowed)
}

///|
/// Add a per-token logit bias. Negative infinity in the base row remains
/// masked, and every bias must be finite. A bias of zero leaves the row alone.
pub fn bias_logits(
  logits : Array[Double],
  biases : Array[Double],
) -> Result[Array[Double], SamplingError] {
  if logits.length() != biases.length() {
    return Err(InvalidParameter("bias length must match vocabulary"))
  }
  match check_logits(logits) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  let output : Array[Double] = []
  for i in 0..<logits.length() {
    if !finite(biases[i]) {
      return Err(InvalidParameter("all logit biases must be finite"))
    }
    let value = logits[i] + biases[i]
    if !finite(value) && value != -1.0 / 0.0 {
      return Err(NumericalFailure)
    }
    output.push(value)
  }
  Ok(output)
}

///|
/// Apply the usual sign-aware repetition penalty once for each token that
/// appears in history. This is a caller-selected preprocessing step, not part
/// of the Mirostat algorithm itself.
pub fn penalize_repetition(
  logits : Array[Double],
  history : Array[Int],
  penalty : Double,
) -> Result[Array[Double], SamplingError] {
  if !finite(penalty) || penalty < 1.0 {
    return Err(InvalidParameter("repetition penalty must be at least one"))
  }
  match check_logits(logits) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  let output = logits.copy()
  let seen = Array::make(logits.length(), false)
  for token in history {
    if token < 0 || token >= logits.length() {
      return Err(InvalidParameter("history token outside vocabulary"))
    }
    if !seen[token] {
      seen[token] = true
      let value = output[token]
      if finite(value) {
        output[token] = if value < 0.0 {
          value * penalty
        } else {
          value / penalty
        }
      }
    }
  }
  Ok(output)
}

///|
/// Subtract an additive presence cost once per seen token and an additive
/// frequency cost for every occurrence. Costs may be negative to reward
/// repetition. This is a preprocessing option outside Mirostat feedback.
pub fn penalize_counts(
  logits : Array[Double],
  history : Array[Int],
  presence : Double,
  frequency : Double,
) -> Result[Array[Double], SamplingError] {
  if !finite(presence) || !finite(frequency) {
    return Err(InvalidParameter("presence and frequency costs must be finite"))
  }
  match check_logits(logits) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  let counts = Array::make(logits.length(), 0)
  for token in history {
    if token < 0 || token >= logits.length() {
      return Err(InvalidParameter("history token outside vocabulary"))
    }
    if counts[token] == 2147483647 {
      return Err(NumericalFailure)
    }
    counts[token] = counts[token] + 1
  }
  let output = logits.copy()
  for token in 0..<output.length() {
    if counts[token] > 0 && finite(output[token]) {
      let adjusted = output[token] -
        presence -
        frequency * counts[token].to_double()
      if !finite(adjusted) {
        return Err(NumericalFailure)
      }
      output[token] = adjusted
    }
  }
  Ok(output)
}
