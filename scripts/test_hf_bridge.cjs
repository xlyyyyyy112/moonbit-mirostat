const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const readline = require("node:readline");
const { once } = require("node:events");

async function main() {
  const child = spawn(
    process.execPath,
    ["integrations/hf_mirostat/bridge.cjs"],
    { cwd: process.cwd(), stdio: ["pipe", "pipe", "inherit"] },
  );
  const output = readline.createInterface({ input: child.stdout });
  async function request(value) {
    const next = once(output, "line");
    child.stdin.write(JSON.stringify(value) + "\n");
    const [line] = await next;
    return JSON.parse(line);
  }

  try {
    const base = {
      logits: [3, 2, 1],
      tau: 2,
      eta: 0.1,
      uniform: 0.2,
      version: "v2",
      state_mu: 4,
      state_steps: 0,
    };
    const first = await request(base);
    assert.equal(first.step.token, 0);
    assert.equal(first.state_steps, 1);
    const second = await request({
      ...base,
      uniform: 0.8,
      state_mu: first.state_mu,
      state_steps: first.state_steps,
    });
    assert.equal(second.state_steps, 2);
    assert.equal(second.step.token, 1);
    const invalid = await request({ ...base, logits: [] });
    assert.equal(invalid.error, true);
    console.log("MoonBit model-host bridge: state, token selection and invalid-row checks passed.");
  } finally {
    output.close();
    child.stdin.end();
    const [code] = await once(child, "exit");
    assert.equal(code, 0);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
