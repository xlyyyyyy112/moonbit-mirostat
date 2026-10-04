///|
/// Shannon entropy of a model row, in bits. Zero-mass tail entries contribute
/// nothing, as in the continuous limit of -p log p.
pub fn entropy(probabilities : Array[Double]) -> Result[Double, SamplingError] {
  if probabilities.length() == 0 {
    return Err(EmptyLogits)
  }
  let mut total = 0.0
  let mut value = 0.0
  for i in 0..<probabilities.length() {
    let p = probabilities[i]
    if !finite(p) || p < 0.0 {
      return Err(InvalidProbability(i))
    }
    total = total + p
    if p > 0.0 {
      value = value - p * @math.ln(p) / @math.ln(2.0)
    }
  }
  if total <= 0.0 || !finite(total) {
    return Err(NumericalFailure)
  }
  Ok(value / total + @math.ln(total) / @math.ln(2.0))
}

///|
/// Original model mass covered by a ranked candidate prefix.
pub fn prefix_mass(
  probabilities : Array[Double],
  order : Array[Int],
  keep : Int,
) -> Result[Double, SamplingError] {
  match check_order(probabilities, order) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  if keep < 1 || keep > order.length() {
    return Err(InvalidParameter("invalid prefix length"))
  }
  let mut mass = 0.0
  for i in 0..<keep {
    mass = mass + probabilities[order[i]]
  }
  if !finite(mass) || mass <= 0.0 {
    return Err(NumericalFailure)
  }
  Ok(mass)
}

///|
/// Expected observed surprise when drawing from a renormalized prefix. The
/// surprise itself is measured against the full original model distribution.
pub fn expected_prefix_surprise(
  probabilities : Array[Double],
  order : Array[Int],
  keep : Int,
) -> Result[Double, SamplingError] {
  let mass = match prefix_mass(probabilities, order, keep) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let mut weighted = 0.0
  for i in 0..<keep {
    let p = probabilities[order[i]]
    if p > 0.0 {
      weighted = weighted - p * @math.ln(p) / @math.ln(2.0)
    }
  }
  Ok(weighted / mass)
}

///|
/// Perplexity corresponding to average surprise measured in bits.
pub fn perplexity(average_surprise : Double) -> Result[Double, SamplingError] {
  if !finite(average_surprise) || average_surprise < 0.0 {
    return Err(InvalidParameter("average surprise must be non-negative"))
  }
  let value = @math.exp(average_surprise * @math.ln(2.0))
  if !finite(value) {
    return Err(NumericalFailure)
  }
  Ok(value)
}

///|
/// Entropy of the renormalized candidate prefix, in bits. It differs from
/// expected model surprise because the latter uses original probabilities.
pub fn prefix_entropy(
  probabilities : Array[Double],
  order : Array[Int],
  keep : Int,
) -> Result[Double, SamplingError] {
  let mass = match prefix_mass(probabilities, order, keep) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let expected = match expected_prefix_surprise(probabilities, order, keep) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  Ok(expected + @math.ln(mass) / @math.ln(2.0))
}

///|
/// Variance of token surprise under the renormalized candidate prefix. This
/// exposes how widely token-level surprises may fluctuate around their mean.
pub fn prefix_surprise_variance(
  probabilities : Array[Double],
  order : Array[Int],
  keep : Int,
) -> Result[Double, SamplingError] {
  let mass = match prefix_mass(probabilities, order, keep) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let mean = match expected_prefix_surprise(probabilities, order, keep) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let mut second_moment = 0.0
  for index in 0..<keep {
    let p = probabilities[order[index]]
    if p > 0.0 {
      let information = -@math.ln(p) / @math.ln(2.0)
      second_moment = second_moment + p * information * information
    }
  }
  let value = second_moment / mass - mean * mean
  if !finite(value) {
    return Err(NumericalFailure)
  }
  Ok(if value < 0.0 { 0.0 } else { value })
}

///|
/// KL divergence in bits from the renormalized prefix distribution to the
/// original distribution. It depends only on retained original probability
/// mass and is zero when the entire support is kept.
pub fn prefix_kl(
  probabilities : Array[Double],
  order : Array[Int],
  keep : Int,
) -> Result[Double, SamplingError] {
  let mass = match prefix_mass(probabilities, order, keep) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  Ok(-@math.ln(mass) / @math.ln(2.0))
}
