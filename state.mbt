///|
/// Portable feedback state for a host that stores Mirostat session data
/// between calls. The configuration and version must be stored separately.
pub struct SamplerState {
  mu : Double
  steps : Int
} derive(Eq, Debug)

///|
pub fn SamplerState::new(
  mu : Double,
  steps : Int,
) -> Result[SamplerState, SamplingError] {
  if !finite(mu) || steps < 0 {
    return Err(InvalidParameter("state needs finite mu and non-negative steps"))
  }
  Ok({ mu, steps, })
}

///|
pub fn SamplerState::mu(self : SamplerState) -> Double {
  self.mu
}

///|
pub fn SamplerState::steps(self : SamplerState) -> Int {
  self.steps
}

///|
pub fn Sampler::state(self : Sampler) -> SamplerState {
  { mu: self.mu, steps: self.steps, }
}

///|
pub fn Sampler::from_state(
  config : Config,
  version : Version,
  state : SamplerState,
) -> Sampler {
  { config, version, mu: state.mu, steps: state.steps, }
}
