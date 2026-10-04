///|
/// A small reproducible RNG for examples, replay fixtures, and portable
/// sampling. Applications may supply uniforms from their own RNG instead.
pub struct ParkMiller {
  mut state : Int
} derive(Debug)

///|
pub fn ParkMiller::new(seed : Int) -> Result[ParkMiller, SamplingError] {
  if seed <= 0 || seed >= 2147483647 {
    return Err(InvalidParameter("seed must be in 1..2147483646"))
  }
  Ok({ state: seed })
}

///|
pub fn ParkMiller::state(self : ParkMiller) -> Int {
  self.state
}

///|
/// Schrage's method computes (16807 * state) mod (2^31 - 1) without an
/// overflowing 32-bit multiplication.
pub fn ParkMiller::next_int(self : ParkMiller) -> Int {
  let high = self.state / 127773
  let low = self.state % 127773
  let candidate = 16807 * low - 2836 * high
  self.state = if candidate > 0 { candidate } else { candidate + 2147483647 }
  self.state
}

///|
/// The result lies strictly inside (0, 1), which satisfies the sampler's
/// required [0, 1) interval.
pub fn ParkMiller::next_uniform(self : ParkMiller) -> Double {
  self.next_int().to_double() / 2147483647.0
}

///|
/// Convenience step using this package's reproducible generator. It consumes
/// one draw even if the supplied logits are invalid; callers requiring exact
/// rollback should validate or manage the RNG separately.
pub fn Sampler::sample_with_rng(
  self : Sampler,
  logits : Array[Double],
  rng : ParkMiller,
) -> Result[Step, SamplingError] {
  self.sample(logits, rng.next_uniform())
}
