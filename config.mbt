///|
/// Mirostat feedback configuration. Tau and mu are measured in bits.
pub struct Config {
  tau : Double
  eta : Double
  initial_mu : Double
  m : Int
} derive(Eq, Debug)

///|
pub fn Config::new(
  tau : Double,
  eta? : Double = 0.1,
  initial_mu? : Double = 2.0 * tau,
  m? : Int = 100,
) -> Result[Config, SamplingError] {
  if !finite(tau) || tau <= 0.0 {
    return Err(InvalidParameter("tau must be finite and positive"))
  }
  if !finite(eta) || eta <= 0.0 {
    return Err(InvalidParameter("eta must be finite and positive"))
  }
  if !finite(initial_mu) || initial_mu < 0.0 {
    return Err(InvalidParameter("initial_mu must be finite and non-negative"))
  }
  if m < 2 {
    return Err(InvalidParameter("m must be at least two"))
  }
  Ok({ tau, eta, initial_mu, m })
}

///|
pub fn Config::tau(self : Config) -> Double {
  self.tau
}

///|
pub fn Config::eta(self : Config) -> Double {
  self.eta
}

///|
pub fn Config::initial_mu(self : Config) -> Double {
  self.initial_mu
}

///|
pub fn Config::m(self : Config) -> Int {
  self.m
}

///|
pub enum Version {
  V1
  V2
} derive(Eq, Debug)

///|
pub fn Version::v1() -> Version {
  V1
}

///|
pub fn Version::v2() -> Version {
  V2
}

///|
pub struct Step {
  token : Int
  original_probability : Double
  observed_surprise : Double
  kept_tokens : Int
  mu_before : Double
  mu_after : Double
} derive(Eq, Debug)

///|
pub fn Step::token(self : Step) -> Int {
  self.token
}

///|
pub fn Step::original_probability(self : Step) -> Double {
  self.original_probability
}

///|
pub fn Step::observed_surprise(self : Step) -> Double {
  self.observed_surprise
}

///|
pub fn Step::kept_tokens(self : Step) -> Int {
  self.kept_tokens
}

///|
pub fn Step::mu_before(self : Step) -> Double {
  self.mu_before
}

///|
pub fn Step::mu_after(self : Step) -> Double {
  self.mu_after
}

///|
/// A session owns only feedback state; the model owns logits and the caller owns RNG.
pub struct Sampler {
  config : Config
  version : Version
  mut mu : Double
  mut steps : Int
} derive(Debug)

///|
pub fn Sampler::new(config : Config, version? : Version = V1) -> Sampler {
  { config, version, mu: config.initial_mu, steps: 0 }
}

///|
pub fn Sampler::mu(self : Sampler) -> Double {
  self.mu
}

///|
pub fn Sampler::steps(self : Sampler) -> Int {
  self.steps
}

///|
pub fn Sampler::version(self : Sampler) -> Version {
  self.version
}

///|
pub fn Sampler::reset(self : Sampler) -> Unit {
  self.mu = self.config.initial_mu
  self.steps = 0
}

///|
fn feedback(mu : Double, observed : Double, config : Config) -> Double {
  mu - config.eta * (observed - config.tau)
}
