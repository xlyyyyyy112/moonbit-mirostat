///|
/// Fixed-size online window for monitoring recent surprise without retaining
/// an entire generation trace. Values are measured in bits.
pub struct SurpriseWindow {
  capacity : Int
  values : Array[Double]
  mut next : Int
  mut count : Int
  mut total : Double
} derive(Debug)

///|
pub fn SurpriseWindow::new(
  capacity : Int,
) -> Result[SurpriseWindow, SamplingError] {
  if capacity < 1 {
    return Err(InvalidParameter("window capacity must be positive"))
  }
  Ok({
    capacity,
    values: Array::make(capacity, 0.0),
    next: 0,
    count: 0,
    total: 0.0,
  })
}

///|
pub fn SurpriseWindow::count(self : SurpriseWindow) -> Int {
  self.count
}

///|
pub fn SurpriseWindow::capacity(self : SurpriseWindow) -> Int {
  self.capacity
}

///|
pub fn SurpriseWindow::push(
  self : SurpriseWindow,
  surprise : Double,
) -> Result[Unit, SamplingError] {
  if !finite(surprise) || surprise < 0.0 {
    return Err(InvalidParameter("surprise must be finite and non-negative"))
  }
  let old = if self.count == self.capacity {
    self.values[self.next]
  } else {
    0.0
  }
  let new_total = self.total - old + surprise
  if !finite(new_total) {
    return Err(NumericalFailure)
  }
  self.values[self.next] = surprise
  self.next = (self.next + 1) % self.capacity
  if self.count < self.capacity {
    self.count = self.count + 1
  }
  self.total = new_total
  Ok(())
}

///|
pub fn SurpriseWindow::push_step(
  self : SurpriseWindow,
  step : Step,
) -> Result[Unit, SamplingError] {
  self.push(step.observed_surprise)
}

///|
pub fn SurpriseWindow::mean(
  self : SurpriseWindow,
) -> Result[Double, SamplingError] {
  if self.count == 0 {
    return Err(InvalidParameter("window has no observations"))
  }
  Ok(self.total / self.count.to_double())
}

///|
pub fn SurpriseWindow::clear(self : SurpriseWindow) -> Unit {
  self.next = 0
  self.count = 0
  self.total = 0.0
}
