///|
/// A single-step JSON request for a host that exchanges model logits with a
/// MoonBit WebAssembly module. The host retains generation session state when
/// using the library API directly; this one-shot protocol starts fresh.
pub struct SamplingRequest {
  logits : Array[Double]
  tau : Double
  eta : Double
  uniform : Double
  version : String
} derive(FromJson, ToJson, Debug)

///|
pub struct SamplingResponse {
  token : Int
  kept_tokens : Int
  probability : Double
  surprise_bits : Double
  mu_after : Double
} derive(FromJson, ToJson, Debug)

///|
pub fn SamplingResponse::token(self : SamplingResponse) -> Int {
  self.token
}

///|
pub fn SamplingResponse::kept_tokens(self : SamplingResponse) -> Int {
  self.kept_tokens
}

///|
pub fn SamplingResponse::surprise_bits(self : SamplingResponse) -> Double {
  self.surprise_bits
}

///|
pub fn SamplingResponse::mu_after(self : SamplingResponse) -> Double {
  self.mu_after
}

///|
fn settings(
  tau : Double,
  eta : Double,
  name : String,
) -> Result[(@mirostat.Config, @mirostat.Version), String] {
  let version = match name {
    "v1" => @mirostat.Version::v1()
    "v2" => @mirostat.Version::v2()
    _ => return Err("version must be v1 or v2")
  }
  let config = match @mirostat.Config::new(tau, eta~) {
    Ok(value) => value
    Err(error) => return Err(error.message())
  }
  Ok((config, version))
}

///|
fn response_for(step : @mirostat.Step) -> SamplingResponse {
  {
    token: step.token(),
    kept_tokens: step.kept_tokens(),
    probability: step.original_probability(),
    surprise_bits: step.observed_surprise(),
    mu_after: step.mu_after(),
  }
}

///|
pub fn sample_request(
  request : SamplingRequest,
) -> Result[SamplingResponse, String] {
  let (config, version) = match
    settings(request.tau, request.eta, request.version) {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let sampler = @mirostat.Sampler::new(config, version~)
  let step = match sampler.sample(request.logits, request.uniform) {
    Ok(value) => value
    Err(error) => return Err(error.message())
  }
  Ok(response_for(step))
}

///|
pub fn sample_json(input : String) -> Result[String, String] {
  let value = @json.parse(input) catch { _ => return Err("invalid JSON") }
  let request : SamplingRequest = @json.from_json(value) catch {
    _ => return Err("invalid sampling request")
  }
  match sample_request(request) {
    Ok(response) => Ok(@json.to_json(response).stringify())
    Err(message) => Err(message)
  }
}

///|
/// A multi-step request. Model logits are precomputed by the host; each row
/// has one matching uniform draw. Feedback state persists across all rows.
pub struct ReplayRequest {
  rows : Array[Array[Double]]
  draws : Array[Double]
  tau : Double
  eta : Double
  version : String
} derive(FromJson, ToJson, Debug)

///|
pub struct ReplayResponse {
  steps : Array[SamplingResponse]
  mean_surprise : Double
  final_mu : Double
} derive(FromJson, ToJson, Debug)

///|
pub fn ReplayResponse::steps(self : ReplayResponse) -> Array[SamplingResponse] {
  self.steps.copy()
}

///|
pub fn ReplayResponse::mean_surprise(self : ReplayResponse) -> Double {
  self.mean_surprise
}

///|
pub fn ReplayResponse::final_mu(self : ReplayResponse) -> Double {
  self.final_mu
}

///|
pub fn replay_request(
  request : ReplayRequest,
) -> Result[ReplayResponse, String] {
  let (config, version) = match
    settings(request.tau, request.eta, request.version) {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let sampler = @mirostat.Sampler::new(config, version~)
  let trace = match
    @mirostat.replay_atomic(sampler, request.rows, request.draws) {
    Ok(value) => value
    Err(error) => return Err(error.message())
  }
  if trace.length() == 0 {
    return Err("replay requires at least one row")
  }
  let summary = match @mirostat.summarize(trace, config.tau()) {
    Ok(value) => value
    Err(error) => return Err(error.message())
  }
  let steps : Array[SamplingResponse] = []
  for step in trace {
    steps.push(response_for(step))
  }
  Ok({
    steps,
    mean_surprise: summary.mean_surprise(),
    final_mu: summary.final_mu(),
  })
}

///|
pub fn replay_json(input : String) -> Result[String, String] {
  let value = @json.parse(input) catch { _ => return Err("invalid JSON") }
  let request : ReplayRequest = @json.from_json(value) catch {
    _ => return Err("invalid replay request")
  }
  match replay_request(request) {
    Ok(response) => Ok(@json.to_json(response).stringify())
    Err(message) => Err(message)
  }
}

///|
/// Stateless wire format for one step of a stateful generation session. A host
/// sends back `state_mu` and `state_steps` from the prior response on its next
/// call, along with the next logit row and random draw.
pub struct SessionRequest {
  logits : Array[Double]
  tau : Double
  eta : Double
  uniform : Double
  version : String
  state_mu : Double
  state_steps : Int
} derive(FromJson, ToJson, Debug)

///|
pub struct SessionResponse {
  step : SamplingResponse
  state_mu : Double
  state_steps : Int
} derive(FromJson, ToJson, Debug)

///|
pub fn SessionResponse::step(self : SessionResponse) -> SamplingResponse {
  self.step
}

///|
pub fn SessionResponse::state_mu(self : SessionResponse) -> Double {
  self.state_mu
}

///|
pub fn SessionResponse::state_steps(self : SessionResponse) -> Int {
  self.state_steps
}

///|
pub fn session_request(
  request : SessionRequest,
) -> Result[SessionResponse, String] {
  let (config, version) = match
    settings(request.tau, request.eta, request.version) {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let state = match
    @mirostat.SamplerState::new(request.state_mu, request.state_steps) {
    Ok(value) => value
    Err(error) => return Err(error.message())
  }
  let sampler = @mirostat.Sampler::from_state(config, version, state)
  let step = match sampler.sample(request.logits, request.uniform) {
    Ok(value) => value
    Err(error) => return Err(error.message())
  }
  Ok({
    step: response_for(step),
    state_mu: sampler.mu(),
    state_steps: sampler.steps(),
  })
}

///|
pub fn session_json(input : String) -> Result[String, String] {
  let value = @json.parse(input) catch { _ => return Err("invalid JSON") }
  let request : SessionRequest = @json.from_json(value) catch {
    _ => return Err("invalid session request")
  }
  match session_request(request) {
    Ok(response) => Ok(@json.to_json(response).stringify())
    Err(message) => Err(message)
  }
}

///|
/// Multi-session JSON request for one batched inference step. Each session
/// retains its own tau, eta, variant, and feedback state.
pub struct BatchRequest {
  sessions : Array[SessionRequest]
} derive(FromJson, ToJson, Debug)

///|
pub struct BatchResponse {
  sessions : Array[SessionResponse]
} derive(FromJson, ToJson, Debug)

///|
pub fn BatchResponse::sessions(self : BatchResponse) -> Array[SessionResponse] {
  self.sessions.copy()
}

///|
pub fn batch_request(request : BatchRequest) -> Result[BatchResponse, String] {
  let samplers : Array[@mirostat.Sampler] = []
  let rows : Array[Array[Double]] = []
  let draws : Array[Double] = []
  for session in request.sessions {
    let (config, version) = match
      settings(session.tau, session.eta, session.version) {
      Ok(value) => value
      Err(message) => return Err(message)
    }
    let state = match
      @mirostat.SamplerState::new(session.state_mu, session.state_steps) {
      Ok(value) => value
      Err(error) => return Err(error.message())
    }
    samplers.push(@mirostat.Sampler::from_state(config, version, state))
    rows.push(session.logits)
    draws.push(session.uniform)
  }
  let steps = match @mirostat.sample_batch_atomic(samplers, rows, draws) {
    Ok(value) => value
    Err(error) => return Err(error.message())
  }
  let sessions : Array[SessionResponse] = []
  for index in 0..<steps.length() {
    sessions.push({
      step: response_for(steps[index]),
      state_mu: samplers[index].mu(),
      state_steps: samplers[index].steps(),
    })
  }
  Ok({ sessions, })
}

///|
pub fn batch_json(input : String) -> Result[String, String] {
  let value = @json.parse(input) catch { _ => return Err("invalid JSON") }
  let request : BatchRequest = @json.from_json(value) catch {
    _ => return Err("invalid batch request")
  }
  match batch_request(request) {
    Ok(response) => Ok(@json.to_json(response).stringify())
    Err(message) => Err(message)
  }
}

///|
/// Inspect the next candidate set without consuming a uniform draw or
/// advancing Mirostat feedback state.
pub struct PreviewRequest {
  logits : Array[Double]
  tau : Double
  eta : Double
  version : String
  state_mu : Double
  state_steps : Int
} derive(FromJson, ToJson, Debug)

///|
pub struct PreviewResponse {
  vocabulary : Int
  kept_tokens : Int
  retained_mass : Double
  expected_surprise : Double
  state_mu : Double
} derive(FromJson, ToJson, Debug)

///|
pub fn PreviewResponse::kept_tokens(self : PreviewResponse) -> Int {
  self.kept_tokens
}

///|
pub fn PreviewResponse::retained_mass(self : PreviewResponse) -> Double {
  self.retained_mass
}

///|
pub fn preview_request(
  request : PreviewRequest,
) -> Result[PreviewResponse, String] {
  let (config, version) = match
    settings(request.tau, request.eta, request.version) {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let state = match
    @mirostat.SamplerState::new(request.state_mu, request.state_steps) {
    Ok(value) => value
    Err(error) => return Err(error.message())
  }
  let sampler = @mirostat.Sampler::from_state(config, version, state)
  let preview = match sampler.preview(request.logits) {
    Ok(value) => value
    Err(error) => return Err(error.message())
  }
  Ok({
    vocabulary: preview.vocabulary(),
    kept_tokens: preview.kept_tokens(),
    retained_mass: preview.retained_mass(),
    expected_surprise: preview.expected_surprise(),
    state_mu: sampler.mu(),
  })
}

///|
pub fn preview_json(input : String) -> Result[String, String] {
  let value = @json.parse(input) catch { _ => return Err("invalid JSON") }
  let request : PreviewRequest = @json.from_json(value) catch {
    _ => return Err("invalid preview request")
  }
  match preview_request(request) {
    Ok(response) => Ok(@json.to_json(response).stringify())
    Err(message) => Err(message)
  }
}

///|
/// One-shot variant for inference backends that expose non-negative weights
/// instead of unnormalized logit scores.
pub struct WeightRequest {
  weights : Array[Double]
  tau : Double
  eta : Double
  uniform : Double
  version : String
} derive(FromJson, ToJson, Debug)

///|
pub fn weight_request(
  request : WeightRequest,
) -> Result[SamplingResponse, String] {
  let (config, version) = match
    settings(request.tau, request.eta, request.version) {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let sampler = @mirostat.Sampler::new(config, version~)
  let step = match sampler.sample_weights(request.weights, request.uniform) {
    Ok(value) => value
    Err(error) => return Err(error.message())
  }
  Ok(response_for(step))
}

///|
pub fn weight_json(input : String) -> Result[String, String] {
  let value = @json.parse(input) catch { _ => return Err("invalid JSON") }
  let request : WeightRequest = @json.from_json(value) catch {
    _ => return Err("invalid weight request")
  }
  match weight_request(request) {
    Ok(response) => Ok(@json.to_json(response).stringify())
    Err(message) => Err(message)
  }
}

///|
/// Start a portable session and obtain the initial feedback state to include
/// in subsequent `session_json` requests.
pub struct InitRequest {
  tau : Double
  eta : Double
  version : String
} derive(FromJson, ToJson, Debug)

///|
pub struct InitResponse {
  state_mu : Double
  state_steps : Int
} derive(FromJson, ToJson, Debug)

///|
pub fn InitResponse::state_mu(self : InitResponse) -> Double {
  self.state_mu
}

///|
pub fn InitResponse::state_steps(self : InitResponse) -> Int {
  self.state_steps
}

///|
pub fn init_request(request : InitRequest) -> Result[InitResponse, String] {
  let (config, version) = match
    settings(request.tau, request.eta, request.version) {
    Ok(value) => value
    Err(message) => return Err(message)
  }
  let sampler = @mirostat.Sampler::new(config, version~)
  Ok({ state_mu: sampler.mu(), state_steps: sampler.steps(), })
}

///|
pub fn init_json(input : String) -> Result[String, String] {
  let value = @json.parse(input) catch { _ => return Err("invalid JSON") }
  let request : InitRequest = @json.from_json(value) catch {
    _ => return Err("invalid init request")
  }
  match init_request(request) {
    Ok(response) => Ok(@json.to_json(response).stringify())
    Err(message) => Err(message)
  }
}
