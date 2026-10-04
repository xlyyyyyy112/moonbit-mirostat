///|
/// Show the one-shot and portable state JSON wire formats without model I/O.
fn main {
  let initial = @jsonio.init_json("{\"tau\":2,\"eta\":0.1,\"version\":\"v2\"}").unwrap()
  println("initial=" + initial)
  let first = @jsonio.session_json(
    "{\"logits\":[3,2,1],\"tau\":2,\"eta\":0.1,\"uniform\":0.2,\"version\":\"v2\",\"state_mu\":4,\"state_steps\":0}",
  ).unwrap()
  println("first_step=" + first)
  let batch = @jsonio.batch_json(
    "{\"sessions\":[{\"logits\":[3,2],\"tau\":2,\"eta\":0.1,\"uniform\":0.2,\"version\":\"v1\",\"state_mu\":4,\"state_steps\":0},{\"logits\":[1,3],\"tau\":2,\"eta\":0.1,\"uniform\":0.8,\"version\":\"v2\",\"state_mu\":4,\"state_steps\":0}]}",
  ).unwrap()
  println("batch=" + batch)
}
