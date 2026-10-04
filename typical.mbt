///|
/// Sort tokens by the distance between their surprise and the model entropy.
/// Zero-probability tokens are placed last, with token id breaking ties.
pub fn typical_order(
  probabilities : Array[Double],
) -> Result[Array[Int], SamplingError] {
  let center = match entropy(probabilities) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let order : Array[Int] = []
  let deviations : Array[Double] = []
  for i in 0..<probabilities.length() {
    order.push(i)
    let p = probabilities[i]
    if p == 0.0 {
      deviations.push(1.0 / 0.0)
    } else {
      let information = -@math.ln(p) / @math.ln(2.0)
      let delta = information - center
      deviations.push(if delta < 0.0 { -delta } else { delta })
    }
  }
  order.sort_by(fn(left, right) {
    if deviations[left] < deviations[right] {
      -1
    } else if deviations[left] > deviations[right] {
      1
    } else {
      left.compare(right)
    }
  })
  Ok(order)
}

///|
/// Choose the smallest locally typical prefix with cumulative mass at least
/// threshold. Unlike top-p, the candidate order need not follow probability.
pub fn typical_prefix_size(
  probabilities : Array[Double],
  order : Array[Int],
  threshold : Double,
) -> Result[Int, SamplingError] {
  if !finite(threshold) || threshold <= 0.0 || threshold > 1.0 {
    return Err(InvalidParameter("typical threshold must be in (0, 1]"))
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
  Ok(order.length())
}

///|
pub fn sample_typical(
  logits : Array[Double],
  threshold : Double,
  uniform : Double,
) -> Result[Int, SamplingError] {
  let p = match probabilities(logits) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let order = match typical_order(p) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let keep = match typical_prefix_size(p, order, threshold) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  draw_prefix(p, order, keep, uniform)
}
