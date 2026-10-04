///|
/// Summarize successive non-overlapping trace segments. The last segment may
/// be shorter. This makes a distribution drift visible instead of hiding it
/// inside one average over the whole run.
pub fn segment_summaries(
  trace : Array[@mirostat.Step],
  target : Double,
  segment_size : Int,
) -> Result[Array[@mirostat.TraceSummary], @mirostat.SamplingError] {
  if segment_size < 1 {
    return Err(
      @mirostat.SamplingError::invalid_parameter(
        "segment size must be positive",
      ),
    )
  }
  let output : Array[@mirostat.TraceSummary] = []
  let mut start = 0
  while start < trace.length() {
    let stop = if start + segment_size < trace.length() {
      start + segment_size
    } else {
      trace.length()
    }
    let segment : Array[@mirostat.Step] = []
    for index in start..<stop {
      segment.push(trace[index])
    }
    match @mirostat.summarize(segment, target) {
      Ok(summary) => output.push(summary)
      Err(error) => return Err(error)
    }
    start = stop
  }
  Ok(output)
}

///|
/// Fraction of generated tokens whose surprise falls within a symmetric
/// tolerance around target. This is a token-level diagnostic; a small hit
/// rate can coexist with an accurate long-run average.
pub fn target_hit_rate(
  trace : Array[@mirostat.Step],
  target : Double,
  tolerance : Double,
) -> Result[Double, @mirostat.SamplingError] {
  if trace.length() == 0 ||
    target <= 0.0 ||
    tolerance < 0.0 ||
    target != target ||
    tolerance != tolerance {
    return Err(
      @mirostat.SamplingError::invalid_parameter("invalid hit-rate inputs"),
    )
  }
  let mut hits = 0
  for step in trace {
    let difference = step.observed_surprise() - target
    let distance = if difference < 0.0 { -difference } else { difference }
    if distance <= tolerance {
      hits = hits + 1
    }
  }
  Ok(hits.to_double() / trace.length().to_double())
}

///|
/// Return a per-segment absolute distance between mean observed surprise and
/// the target. Unlike mean absolute token error, this measures local drift.
pub fn segment_mean_errors(
  trace : Array[@mirostat.Step],
  target : Double,
  segment_size : Int,
) -> Result[Array[Double], @mirostat.SamplingError] {
  let summaries = match segment_summaries(trace, target, segment_size) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  let errors : Array[Double] = []
  for summary in summaries {
    let difference = summary.mean_surprise() - target
    errors.push(if difference < 0.0 { -difference } else { difference })
  }
  Ok(errors)
}
