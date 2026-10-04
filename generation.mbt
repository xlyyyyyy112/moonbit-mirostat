///|
pub enum StopReason {
  Limit
  EndToken
  Observer
} derive(Eq, Debug)

///|
pub fn StopReason::limit() -> StopReason {
  Limit
}

///|
pub fn StopReason::end_token() -> StopReason {
  EndToken
}

///|
pub fn StopReason::observer() -> StopReason {
  Observer
}

///|
pub struct Generation {
  tokens : Array[Int]
  steps : Array[Step]
  reason : StopReason
} derive(Eq, Debug)

///|
pub fn Generation::tokens(self : Generation) -> Array[Int] {
  self.tokens.copy()
}

///|
pub fn Generation::steps(self : Generation) -> Array[Step] {
  self.steps.copy()
}

///|
pub fn Generation::reason(self : Generation) -> StopReason {
  self.reason
}

///|
/// Drive an autoregressive model callback until an end token or step limit.
/// The callback receives a copy of the generated prefix and must return the
/// next logit row. The RNG is consumed only after a model row is available.
/// Any error preserves successful prior sampler steps and RNG draws.
pub fn generate(
  sampler : Sampler,
  rng : ParkMiller,
  model : (Array[Int]) -> Result[Array[Double], SamplingError],
  limit : Int,
  end_token : Int?,
) -> Result[Generation, SamplingError] {
  generate_from_prompt(sampler, rng, model, [], limit, end_token)
}

///|
/// Generate after a pre-existing prompt. The callback sees prompt followed by
/// generated tokens; the returned token list contains only new tokens.
pub fn generate_from_prompt(
  sampler : Sampler,
  rng : ParkMiller,
  model : (Array[Int]) -> Result[Array[Double], SamplingError],
  prompt : Array[Int],
  limit : Int,
  end_token : Int?,
) -> Result[Generation, SamplingError] {
  let steps : Array[Step] = []
  let result = match
    generate_stream(sampler, rng, model, prompt, limit, end_token, fn(step) {
      steps.push(step)
      true
    }) {
    Ok(value) => value
    Err(error) => return Err(error)
  }
  Ok({ tokens: result.tokens(), steps, reason: result.reason(), })
}

///|
pub struct StreamGeneration {
  tokens : Array[Int]
  reason : StopReason
} derive(Eq, Debug)

///|
pub fn StreamGeneration::tokens(self : StreamGeneration) -> Array[Int] {
  self.tokens.copy()
}

///|
pub fn StreamGeneration::count(self : StreamGeneration) -> Int {
  self.tokens.length()
}

///|
pub fn StreamGeneration::reason(self : StreamGeneration) -> StopReason {
  self.reason
}

///|
/// Stream each successful decision to an observer and avoid retaining Step
/// metadata. Return false from the observer to stop after that token. End
/// token detection takes priority when both conditions apply to one step.
pub fn generate_stream(
  sampler : Sampler,
  rng : ParkMiller,
  model : (Array[Int]) -> Result[Array[Double], SamplingError],
  prompt : Array[Int],
  limit : Int,
  end_token : Int?,
  observer : (Step) -> Bool,
) -> Result[StreamGeneration, SamplingError] {
  if limit < 0 {
    return Err(InvalidParameter("generation limit must be non-negative"))
  }
  match end_token {
    Some(token) =>
      if token < 0 {
        return Err(InvalidParameter("end token must be non-negative"))
      }
    None => ()
  }
  for token in prompt {
    if token < 0 {
      return Err(InvalidParameter("prompt token must be non-negative"))
    }
  }
  let tokens : Array[Int] = []
  for _ in 0..<limit {
    let context = prompt.copy()
    for token in tokens {
      context.push(token)
    }
    let logits = match model(context) {
      Ok(value) => value
      Err(error) => return Err(error)
    }
    let step = match sampler.sample_with_rng(logits, rng) {
      Ok(value) => value
      Err(error) => return Err(error)
    }
    tokens.push(step.token())
    let keep_going = observer(step)
    match end_token {
      Some(token) =>
        if step.token() == token {
          return Ok({ tokens, reason: EndToken, })
        }
      None => ()
    }
    if !keep_going {
      return Ok({ tokens, reason: Observer, })
    }
  }
  Ok({ tokens, reason: Limit, })
}
