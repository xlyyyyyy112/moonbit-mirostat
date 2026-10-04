///|
/// A snapshot of one sampler's feedback state and configuration. A checkpoint
/// can be restored only to a sampler with the same configuration and variant.
pub struct Checkpoint {
  config : Config
  version : Version
  mu : Double
  steps : Int
} derive(Eq, Debug)

///|
pub fn Checkpoint::mu(self : Checkpoint) -> Double {
  self.mu
}

///|
pub fn Checkpoint::steps(self : Checkpoint) -> Int {
  self.steps
}

///|
pub fn Sampler::checkpoint(self : Sampler) -> Checkpoint {
  { config: self.config, version: self.version, mu: self.mu, steps: self.steps }
}

///|
pub fn Sampler::restore(
  self : Sampler,
  checkpoint : Checkpoint,
) -> Result[Unit, SamplingError] {
  if self.config != checkpoint.config || self.version != checkpoint.version {
    return Err(InvalidParameter("checkpoint belongs to a different sampler"))
  }
  self.mu = checkpoint.mu
  self.steps = checkpoint.steps
  Ok(())
}

///|
/// Copy a sampler's feedback state into a separate mutable session. This
/// supports branching generation without sharing subsequent feedback.
pub fn Sampler::fork(self : Sampler) -> Sampler {
  { config: self.config, version: self.version, mu: self.mu, steps: self.steps }
}
