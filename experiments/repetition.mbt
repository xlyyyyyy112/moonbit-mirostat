///|
/// Fraction of n-gram windows that repeat an earlier window in the same token
/// sequence. This measures one property discussed in the Mirostat paper, but
/// it cannot by itself judge meaning or text quality.
pub fn repeated_ngram_rate(
  tokens : Array[Int],
  n : Int,
) -> Result[Double, @mirostat.SamplingError] {
  if n < 1 || n > tokens.length() {
    return Err(
      @mirostat.SamplingError::invalid_parameter(
        "n must fit the token sequence",
      ),
    )
  }
  let windows = tokens.length() - n + 1
  let mut repeated = 0
  for start in 0..<windows {
    let mut seen = false
    for earlier in 0..<start {
      if same_window(tokens, earlier, start, n) {
        seen = true
        break
      }
    }
    if seen {
      repeated = repeated + 1
    }
  }
  Ok(repeated.to_double() / windows.to_double())
}

///|
fn same_window(tokens : Array[Int], left : Int, right : Int, n : Int) -> Bool {
  for offset in 0..<n {
    if tokens[left + offset] != tokens[right + offset] {
      return false
    }
  }
  true
}

///|
/// Fraction of unique token ids in a generated sequence. This statistic is
/// descriptive only; a high value does not guarantee coherent text.
pub fn distinct_token_rate(
  tokens : Array[Int],
) -> Result[Double, @mirostat.SamplingError] {
  if tokens.length() == 0 {
    return Err(
      @mirostat.SamplingError::invalid_parameter("tokens must not be empty"),
    )
  }
  let mut distinct = 0
  for index in 0..<tokens.length() {
    let mut seen = false
    for earlier in 0..<index {
      if tokens[earlier] == tokens[index] {
        seen = true
        break
      }
    }
    if !seen {
      distinct = distinct + 1
    }
  }
  Ok(distinct.to_double() / tokens.length().to_double())
}

///|
pub fn trace_tokens(trace : Array[@mirostat.Step]) -> Array[Int] {
  let tokens : Array[Int] = []
  for step in trace {
    tokens.push(step.token())
  }
  tokens
}
