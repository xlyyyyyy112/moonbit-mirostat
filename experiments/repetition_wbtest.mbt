///|
test "repeated ngrams count windows with earlier matches" {
  assert_eq(repeated_ngram_rate([1, 2, 1, 2], 2), Ok(1.0 / 3.0))
  assert_eq(repeated_ngram_rate([1, 2, 3], 2), Ok(0.0))
  assert_eq(repeated_ngram_rate([1, 1, 1], 1), Ok(2.0 / 3.0))
}

///|
test "distinct token rate counts each id once" {
  assert_eq(distinct_token_rate([1, 2, 1, 3]), Ok(0.75))
  assert_true(distinct_token_rate([]) is Err(_))
  assert_true(repeated_ngram_rate([1], 2) is Err(_))
}
