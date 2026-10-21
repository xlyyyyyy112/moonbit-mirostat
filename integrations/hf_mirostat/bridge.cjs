const readline = require("node:readline");
const { session_json } = require("../../_build/js/release/build/bridge/bridge.js");

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
async function main() {
  for await (const line of input) {
    try {
      process.stdout.write(session_json(line) + "\n");
    } catch (error) {
      process.stdout.write(JSON.stringify({ error: String(error) }) + "\n");
    }
  }
}
main().catch((error) => {
  process.stderr.write(String(error) + "\n");
  process.exitCode = 1;
});
