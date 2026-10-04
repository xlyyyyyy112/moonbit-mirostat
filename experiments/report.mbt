///|
/// Render a two-variant comparison as CSV. All fields are numeric except the
/// variant label, so no CSV escaping is required.
pub fn comparison_csv(comparison : Comparison, target : Double) -> String {
  let first = comparison.v1()
  let second = comparison.v2()
  "variant,target,mean_surprise,mean_absolute_error,mean_kept,final_mu\n" +
  "v1," +
  target.to_string() +
  "," +
  first.mean_surprise().to_string() +
  "," +
  first.mean_absolute_error().to_string() +
  "," +
  first.mean_kept().to_string() +
  "," +
  first.final_mu().to_string() +
  "\n" +
  "v2," +
  target.to_string() +
  "," +
  second.mean_surprise().to_string() +
  "," +
  second.mean_absolute_error().to_string() +
  "," +
  second.mean_kept().to_string() +
  "," +
  second.final_mu().to_string() +
  "\n"
}

///|
/// Render a target sweep in the same row order as the supplied targets.
pub fn sweep_csv(points : Array[SweepPoint]) -> String {
  let mut output = "target,v1_mean,v1_error,v2_mean,v2_error\n"
  for point in points {
    let comparison = point.comparison()
    output = output +
      point.target().to_string() +
      "," +
      comparison.v1().mean_surprise().to_string() +
      "," +
      point.v1_error().to_string() +
      "," +
      comparison.v2().mean_surprise().to_string() +
      "," +
      point.v2_error().to_string() +
      "\n"
  }
  output
}
