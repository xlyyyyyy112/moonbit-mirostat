///|
test "online window evicts oldest surprise" {
  let window = SurpriseWindow::new(2).unwrap()
  assert_true(window.mean() is Err(_))
  window.push(1.0).unwrap()
  window.push(3.0).unwrap()
  assert_eq(window.mean(), Ok(2.0))
  window.push(5.0).unwrap()
  assert_eq(window.mean(), Ok(4.0))
  assert_eq(window.count(), 2)
  window.clear()
  assert_eq(window.count(), 0)
}

///|
test "online window rejects invalid observations" {
  let window = SurpriseWindow::new(3).unwrap()
  assert_true(window.push(-1.0) is Err(_))
  assert_true(SurpriseWindow::new(0) is Err(_))
  assert_eq(window.count(), 0)
}
