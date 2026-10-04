///|
/// Stateless top-k sampling is useful as a controlled comparison for Mirostat.
pub fn sample_top_k(
  logits : Array[Double],
  k : Int,
  uniform : Double,
) -> Result[Int, SamplingError] {
  let p = match probabilities(logits) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  if k < 1 || k > p.length() {
    return Err(InvalidParameter("k must be between one and vocabulary size"))
  }
  let order = match top_indices(p, k) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  draw_candidates(p, order, uniform)
}

///|
/// Return the smallest top-probability prefix with mass at least threshold.
pub fn top_p_prefix_size(
  probabilities : Array[Double],
  order : Array[Int],
  threshold : Double,
) -> Result[Int, SamplingError] {
  if !finite(threshold) ||
    threshold <= 0.0 ||
    threshold > 1.0 ||
    probabilities.length() == 0 ||
    probabilities.length() != order.length() {
    return Err(InvalidParameter("top-p threshold must be in (0, 1]"))
  }
  match check_order(probabilities, order) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  let mut mass = 0.0
  for i in 0..<order.length() {
    mass = mass + probabilities[order[i]]
    if mass >= threshold {
      return Ok(i + 1)
    }
  }
  // Floating-point softmax sums may be just below one.
  Ok(order.length())
}

///|
/// Stateless nucleus sampling on the original model distribution.
pub fn sample_top_p(
  logits : Array[Double],
  threshold : Double,
  uniform : Double,
) -> Result[Int, SamplingError] {
  let p = match probabilities(logits) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let order = match rank(p) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let keep = match top_p_prefix_size(p, order, threshold) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  draw_prefix(p, order, keep, uniform)
}

///|
/// Scale logits by positive temperature before another sampling operation.
pub fn with_temperature(
  logits : Array[Double],
  temperature : Double,
) -> Result[Array[Double], SamplingError] {
  match check_logits(logits) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  if !finite(temperature) || temperature <= 0.0 {
    return Err(InvalidParameter("temperature must be finite and positive"))
  }
  // Subtracting the common maximum leaves softmax unchanged and avoids
  // overflow when a very small temperature scales large model logits.
  let mut maximum = logits[0]
  for value in logits {
    if value > maximum {
      maximum = value
    }
  }
  let scaled : Array[Double] = []
  for value in logits {
    let result = (value - maximum) / temperature
    if !finite(result) && result != -1.0 / 0.0 {
      return Err(NumericalFailure)
    }
    scaled.push(result)
  }
  Ok(scaled)
}

///|
/// Independent temperature sampling without truncation.
pub fn sample_temperature(
  logits : Array[Double],
  temperature : Double,
  uniform : Double,
) -> Result[Int, SamplingError] {
  let scaled = match with_temperature(logits, temperature) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  sample_top_k(scaled, scaled.length(), uniform)
}
