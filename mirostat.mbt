///|
/// Estimate the Zipf exponent from adjacent probability ratios in the top m.
/// This follows the paper's least-squares estimator and uses log base two.
pub fn zipf_exponent(
  probabilities : Array[Double],
  order : Array[Int],
  m : Int,
) -> Result[Double, SamplingError] {
  if m < 2 || order.length() != probabilities.length() || order.length() < 2 {
    return Err(InvalidParameter("Zipf estimation needs two ranked tokens"))
  }
  match check_order(probabilities, order) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  estimate_zipf_prefix(probabilities, order, m)
}

///|
fn estimate_zipf_prefix(
  probabilities : Array[Double],
  order : Array[Int],
  m : Int,
) -> Result[Double, SamplingError] {
  let pairs = if m < order.length() { m - 1 } else { order.length() - 1 }
  let mut numerator = 0.0
  let mut denominator = 0.0
  for i in 0..<pairs {
    let left = probabilities[order[i]]
    let right = probabilities[order[i + 1]]
    if left <= 0.0 || right <= 0.0 {
      break
    }
    let rank_ratio = @math.ln((i + 2).to_double() / (i + 1).to_double())
    let probability_ratio = @math.ln(left) - @math.ln(right)
    numerator = numerator + rank_ratio * probability_ratio
    denominator = denominator + rank_ratio * rank_ratio
  }
  if denominator <= 0.0 {
    return Ok(0.000001)
  }
  let s = numerator / denominator
  if !finite(s) {
    return Err(NumericalFailure)
  }
  // A flat distribution has slope zero. The limiting positive slope keeps
  // the adaptive-k expression defined while preserving its flatness.
  Ok(if s < 0.000001 { 0.000001 } else { s })
}

///|
/// The paper's adaptive top-k estimate. Near s=1 the limiting expression
/// uses ln(vocabulary size), avoiding a 0/0 numerical failure.
pub fn estimated_k(
  mu : Double,
  exponent : Double,
  vocabulary : Int,
) -> Result[Int, SamplingError] {
  if !finite(mu) || !finite(exponent) || exponent <= 0.0 || vocabulary <= 0 {
    return Err(InvalidParameter("invalid Mirostat k inputs"))
  }
  if vocabulary == 1 {
    return Ok(1)
  }
  let epsilon = exponent - 1.0
  let denominator = if epsilon > -0.000001 && epsilon < 0.000001 {
    @math.ln(vocabulary.to_double())
  } else {
    (1.0 - @math.exp(-epsilon * @math.ln(vocabulary.to_double()))) / epsilon
  }
  if !finite(denominator) || denominator <= 0.0 {
    return Err(NumericalFailure)
  }
  let log_k = (mu * @math.ln(2.0) - @math.ln(denominator)) / exponent
  if !finite(log_k) {
    return Err(NumericalFailure)
  }
  if log_k <= 0.0 {
    return Ok(1)
  }
  if log_k >= @math.ln(vocabulary.to_double()) {
    return Ok(vocabulary)
  }
  let raw = @math.exp(log_k)
  let rounded = (raw + 0.5).to_int()
  Ok(
    if rounded < 1 {
      1
    } else if rounded > vocabulary {
      vocabulary
    } else {
      rounded
    },
  )
}

///|
/// V2 uses the current surprise target as a cutoff on original token
/// probabilities. It always keeps at least the most likely token.
pub fn v2_prefix_size(
  probabilities : Array[Double],
  order : Array[Int],
  mu : Double,
) -> Result[Int, SamplingError] {
  if !finite(mu) ||
    order.length() == 0 ||
    order.length() != probabilities.length() {
    return Err(InvalidParameter("invalid V2 prefix inputs"))
  }
  match check_order(probabilities, order) {
    Ok(_) => ()
    Err(error) => return Err(error)
  }
  let mut kept = 0
  for token in order {
    let p = probabilities[token]
    if p <= 0.0 {
      break
    }
    let information = match surprise(p) {
      Ok(value) => value
      Err(e) => return Err(e)
    }
    if information <= mu {
      kept = kept + 1
    } else {
      break
    }
  }
  Ok(if kept == 0 { 1 } else { kept })
}

///|
/// One complete Mirostat step. State changes only after every calculation
/// succeeds, so malformed logits cannot corrupt an ongoing generation.
pub fn Sampler::sample(
  self : Sampler,
  logits : Array[Double],
  uniform : Double,
) -> Result[Step, SamplingError] {
  let p = match probabilities(logits) {
    Ok(value) => value
    Err(e) => return Err(e)
  }
  self.sample_distribution(p, uniform)
}

///|
/// Sample from an already computed model distribution, supplied as weights.
/// The weights are normalized internally and need not sum to one.
pub fn Sampler::sample_weights(
  self : Sampler,
  weights : Array[Double],
  uniform : Double,
) -> Result[Step, SamplingError] {
  let p = match normalize_weights(weights) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  self.sample_distribution(p, uniform)
}

///|
fn Sampler::sample_distribution(
  self : Sampler,
  p : Array[Double],
  uniform : Double,
) -> Result[Step, SamplingError] {
  let order = match self.select_candidates(p) {
    Ok(value) => value
    Err(e) => return Err(e)
  }
  let token = match draw_candidates(p, order, uniform) {
    Ok(value) => value
    Err(e) => return Err(e)
  }
  self.commit_step(p, token, order.length())
}

///|
fn Sampler::commit_step(
  self : Sampler,
  p : Array[Double],
  token : Int,
  keep : Int,
) -> Result[Step, SamplingError] {
  let observed = match surprise(p[token]) {
    Ok(value) => value
    Err(e) => return Err(e)
  }
  let before = self.mu
  let after = feedback(before, observed, self.config)
  if !finite(after) {
    return Err(NumericalFailure)
  }
  self.mu = after
  self.steps = self.steps + 1
  Ok({
    token,
    original_probability: p[token],
    observed_surprise: observed,
    kept_tokens: keep,
    mu_before: before,
    mu_after: after,
  })
}

///|
fn Sampler::select_candidates(
  self : Sampler,
  p : Array[Double],
) -> Result[Array[Int], SamplingError] {
  match self.version {
    V1 =>
      if p.length() == 1 {
        Ok([0])
      } else {
        let m = if self.config.m < p.length() {
          self.config.m
        } else {
          p.length()
        }
        let top_m = match top_indices(p, m) {
          Ok(value) => value
          Err(error) => return Err(error)
        }
        let s = match estimate_zipf_prefix(p, top_m, m) {
          Ok(value) => value
          Err(e) => return Err(e)
        }
        let k = match estimated_k(self.mu, s, p.length()) {
          Ok(value) => value
          Err(error) => return Err(error)
        }
        top_indices(p, k)
      }
    V2 => {
      let candidates : Array[Int] = []
      for token in 0..<p.length() {
        if p[token] > 0.0 {
          let information = match surprise(p[token]) {
            Ok(value) => value
            Err(error) => return Err(error)
          }
          if information <= self.mu {
            candidates.push(token)
          }
        }
      }
      if candidates.length() == 0 {
        return top_indices(p, 1)
      }
      candidates.sort_by(fn(left, right) {
        if better_token(left, right, p) {
          -1
        } else if better_token(right, left, p) {
          1
        } else {
          0
        }
      })
      Ok(candidates)
    }
  }
}
