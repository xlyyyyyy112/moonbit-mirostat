///|
/// Return the k highest-probability token indices in descending order.
/// A size-k heap avoids sorting the full vocabulary when k is small.
pub fn top_indices(
  probabilities : Array[Double],
  k : Int,
) -> Result[Array[Int], SamplingError] {
  if probabilities.length() == 0 {
    return Err(EmptyLogits)
  }
  if k < 1 || k > probabilities.length() {
    return Err(InvalidParameter("k must be between one and vocabulary size"))
  }
  let heap : Array[Int] = []
  for token in 0..<probabilities.length() {
    let p = probabilities[token]
    if !finite(p) || p < 0.0 {
      return Err(InvalidProbability(token))
    }
    if heap.length() < k {
      heap.push(token)
      sift_up(heap, probabilities, heap.length() - 1)
    } else if better_token(token, heap[0], probabilities) {
      heap[0] = token
      sift_down(heap, probabilities, 0)
    }
  }
  heap.sort_by(fn(left, right) {
    if better_token(left, right, probabilities) {
      -1
    } else if better_token(right, left, probabilities) {
      1
    } else {
      0
    }
  })
  Ok(heap)
}

///|
fn better_token(left : Int, right : Int, p : Array[Double]) -> Bool {
  p[left] > p[right] || (p[left] == p[right] && left < right)
}

///|
fn worse_token(left : Int, right : Int, p : Array[Double]) -> Bool {
  better_token(right, left, p)
}

///|
fn sift_up(heap : Array[Int], p : Array[Double], index : Int) -> Unit {
  let mut child = index
  while child > 0 {
    let parent = (child - 1) / 2
    if !worse_token(heap[child], heap[parent], p) {
      break
    }
    let old = heap[parent]
    heap[parent] = heap[child]
    heap[child] = old
    child = parent
  }
}

///|
fn sift_down(heap : Array[Int], p : Array[Double], index : Int) -> Unit {
  let mut parent = index
  while true {
    let left = parent * 2 + 1
    if left >= heap.length() {
      break
    }
    let right = left + 1
    let mut worst = left
    if right < heap.length() && worse_token(heap[right], heap[left], p) {
      worst = right
    }
    if !worse_token(heap[worst], heap[parent], p) {
      break
    }
    let old = heap[parent]
    heap[parent] = heap[worst]
    heap[worst] = old
    parent = worst
  }
}
