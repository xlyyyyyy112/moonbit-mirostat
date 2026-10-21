///|
/// Process one stateful sampling request from the Python model host.
/// Errors use the same JSON string wire protocol as jsonio.
#export_name("session_json")
pub fn session_json(input : String) -> String {
  match @jsonio.session_json(input) {
    Ok(output) => output
    Err(_) => "{\"error\":true}"
  }
}
