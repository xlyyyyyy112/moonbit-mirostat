///|
test "allow mask removes forbidden token" {
  let output = mask_logits([3.0, 2.0, 1.0], [false, true, false]).unwrap()
  assert_eq(sample_top_k(output, 1, 0.5), Ok(1))
  assert_eq(mask_logits([1.0], [false]), Err(AllMasked))
}

///|
test "bias changes ranking without changing source" {
  let source = [2.0, 1.0]
  let changed = bias_logits(source, [0.0, 2.0]).unwrap()
  assert_eq(sample_top_k(changed, 1, 0.5), Ok(1))
  assert_eq(source[1], 1.0)
}

///|
test "repetition penalty is applied only once per distinct token" {
  let output = penalize_repetition([4.0, -2.0, 1.0], [0, 0, 1], 2.0).unwrap()
  assert_eq(output, [2.0, -4.0, 1.0])
}

///|
test "processors reject inconsistent arrays and history" {
  assert_true(mask_logits([0.0], []) is Err(_))
  assert_true(bias_logits([0.0], []) is Err(_))
  assert_true(penalize_repetition([0.0], [1], 1.5) is Err(_))
}

///|
test "additive count penalties use occurrences" {
  let row = penalize_counts([5.0, 4.0, 3.0], [0, 0, 1], 1.0, 0.5).unwrap()
  assert_eq(row, [3.0, 2.5, 3.0])
  assert_true(penalize_counts([0.0], [1], 1.0, 1.0) is Err(_))
}

///|
test "preprocessing retains masked entries" {
  let masked = -1.0 / 0.0
  let source = [3.0, masked, 1.0]
  let by_count = penalize_counts(source, [1, 1], 1.0, 0.5).unwrap()
  let by_repeat = penalize_repetition(by_count, [1], 2.0).unwrap()
  let by_bias = bias_logits(by_repeat, [0.0, 2.0, 0.0]).unwrap()
  assert_eq(probabilities(by_bias).unwrap()[1], 0.0)
}

///|
test "sparse token masks match full Boolean masks" {
  let logits = [4.0, 3.0, 2.0, 1.0]
  let blocked = block_token_ids(logits, [0, 2, 2]).unwrap()
  let allowed = allow_token_ids(logits, [1, 3]).unwrap()
  let full = mask_logits(logits, [false, true, false, true]).unwrap()
  assert_eq(blocked, full)
  assert_eq(allowed, full)
  assert_true(allow_token_ids(logits, []) is Err(AllMasked))
  assert_true(block_token_ids(logits, [4]) is Err(_))
}

///|
test "sparse allowlist integrates with adaptive sampling" {
  let logits = allow_token_ids([5.0, 4.0, 3.0, 2.0], [1, 3]).unwrap()
  let config = Config::new(2.0).unwrap()
  for version in [Version::v1(), Version::v2()] {
    let sampler = Sampler::new(config, version~)
    for draw in [0.0, 0.5, 0.99] {
      let step = sampler.sample(logits, draw).unwrap()
      assert_true(step.token() == 1 || step.token() == 3)
      assert_true(step.original_probability() > 0.0)
    }
    assert_eq(sampler.steps(), 3)
    assert_true(sampler.preview(logits).unwrap().retained_mass() > 0.0)
  }
}
