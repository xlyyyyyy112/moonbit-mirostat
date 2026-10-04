///|
test "one-shot JSON sampling returns chosen token" {
  let response = sample_json(
    "{\"logits\":[3,2,1],\"tau\":2,\"eta\":0.1,\"uniform\":0.2,\"version\":\"v2\"}",
  ).unwrap()
  let decoded : SamplingResponse = @json.from_json(@json.parse(response))
  assert_eq(decoded.token(), 0)
  assert_true(decoded.kept_tokens() >= 1)
  assert_true(decoded.surprise_bits() >= 0.0)
}

///|
test "JSON protocol reports malformed input" {
  assert_eq(sample_json("{"), Err("invalid JSON"))
  assert_eq(sample_json("{}"), Err("invalid sampling request"))
  assert_true(
    sample_json(
      "{\"logits\":[1],\"tau\":2,\"eta\":0.1,\"uniform\":0.2,\"version\":\"v3\"}",
    )
    is Err(_),
  )
}

///|
test "JSON replay preserves feedback across rows" {
  let response = replay_json(
    "{\"rows\":[[3,2,1],[2,1,0]],\"draws\":[0.2,0.8],\"tau\":2,\"eta\":0.1,\"version\":\"v2\"}",
  ).unwrap()
  let decoded : ReplayResponse = @json.from_json(@json.parse(response))
  assert_eq(decoded.steps().length(), 2)
  assert_true(decoded.mean_surprise() >= 0.0)
  assert_eq(decoded.final_mu(), decoded.steps()[1].mu_after())
}

///|
test "JSON session can be continued across requests" {
  let first = session_json(
    "{\"logits\":[3,2,1],\"tau\":2,\"eta\":0.1,\"uniform\":0.2,\"version\":\"v2\",\"state_mu\":4,\"state_steps\":0}",
  ).unwrap()
  let first : SessionResponse = @json.from_json(@json.parse(first))
  assert_eq(first.state_steps(), 1)
  let second = session_request({
    logits: [2.0, 1.0, 0.0],
    tau: 2.0,
    eta: 0.1,
    uniform: 0.8,
    version: "v2",
    state_mu: first.state_mu(),
    state_steps: first.state_steps(),
  }).unwrap()
  assert_eq(second.state_steps(), 2)
  assert_eq(second.state_mu(), second.step().mu_after())
}

///|
test "JSON batch keeps session states independent" {
  let response = batch_json(
    "{\"sessions\":[{\"logits\":[3,2],\"tau\":2,\"eta\":0.1,\"uniform\":0.1,\"version\":\"v1\",\"state_mu\":4,\"state_steps\":0},{\"logits\":[1,3],\"tau\":3,\"eta\":0.1,\"uniform\":0.9,\"version\":\"v2\",\"state_mu\":6,\"state_steps\":5}]}",
  ).unwrap()
  let decoded : BatchResponse = @json.from_json(@json.parse(response))
  assert_eq(decoded.sessions().length(), 2)
  assert_eq(decoded.sessions()[0].state_steps(), 1)
  assert_eq(decoded.sessions()[1].state_steps(), 6)
}

///|
test "JSON preview keeps original feedback state" {
  let response = preview_json(
    "{\"logits\":[3,2,1],\"tau\":2,\"eta\":0.1,\"version\":\"v2\",\"state_mu\":4,\"state_steps\":5}",
  ).unwrap()
  let decoded : PreviewResponse = @json.from_json(@json.parse(response))
  assert_true(decoded.kept_tokens() >= 1)
  assert_true(decoded.retained_mass() > 0.0)
}

///|
test "JSON weight request normalizes model values" {
  let response = weight_json(
    "{\"weights\":[8,4,2,1],\"tau\":2,\"eta\":0.1,\"uniform\":0.2,\"version\":\"v2\"}",
  ).unwrap()
  let decoded : SamplingResponse = @json.from_json(@json.parse(response))
  assert_eq(decoded.token(), 0)
}

///|
test "JSON initialization returns paper default mu" {
  let response = init_json("{\"tau\":3,\"eta\":0.1,\"version\":\"v1\"}").unwrap()
  let decoded : InitResponse = @json.from_json(@json.parse(response))
  assert_eq(decoded.state_mu(), 6.0)
  assert_eq(decoded.state_steps(), 0)
}
