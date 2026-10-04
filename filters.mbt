///|
/// Keep every token whose original model probability is at least a fraction
/// of the best token's probability. The best token is always kept.
pub fn min_p_prefix_size(
  probabilities : Array[Double],
  order : Array[Int],
  min_p : Double,
) -> Result[Int, SamplingError] {
  if !finite(min_p) || min_p < 0.0 || min_p > 1.0 {
    return Err(InvalidParameter("min-p must be in [0, 1]"))
  }
  match check_order(probabilities, order) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  let cutoff = probabilities[order[0]] * min_p
  let mut keep = 0
  for token in order {
    if probabilities[token] < cutoff {
      break
    }
    keep = keep + 1
  }
  Ok(if keep < 1 { 1 } else { keep })
}

///|
/// Sample from a min-p filtered model row.
pub fn sample_min_p(
  logits : Array[Double],
  min_p : Double,
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
  let keep = match min_p_prefix_size(p, order, min_p) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  draw_prefix(p, order, keep, uniform)
}

///|
/// Keep the ranked prefix whose tokens each have probability at least the
/// absolute epsilon threshold. At least the best token survives.
pub fn epsilon_prefix_size(
  probabilities : Array[Double],
  order : Array[Int],
  epsilon : Double,
) -> Result[Int, SamplingError] {
  if !finite(epsilon) || epsilon < 0.0 || epsilon > 1.0 {
    return Err(InvalidParameter("epsilon must be in [0, 1]"))
  }
  match check_order(probabilities, order) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  let mut keep = 0
  for token in order {
    if probabilities[token] < epsilon {
      break
    }
    keep = keep + 1
  }
  Ok(if keep < 1 { 1 } else { keep })
}

///|
pub fn sample_epsilon(
  logits : Array[Double],
  epsilon : Double,
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
  let keep = match epsilon_prefix_size(p, order, epsilon) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  draw_prefix(p, order, keep, uniform)
}
