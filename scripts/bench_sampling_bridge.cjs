"use strict";

const { performance } = require("node:perf_hooks");
const { session_json } = require("../_build/js/release/build/bridge/bridge.js");

const vocabulary = 50257;
const rowsPerRun = 24;
const repeats = 5;
const logits = Array(vocabulary).fill(0);

function request(version) {
  return JSON.stringify({
    logits,
    tau: 4,
    eta: 0.1,
    uniform: 0.375,
    version,
    state_mu: 20,
    state_steps: 0,
  });
}

function batch(input) {
  for (let i = 0; i < rowsPerRun; i++) {
    session_json(input);
  }
}

console.log(`node=${process.version}, vocab=${vocabulary}, rows/run=${rowsPerRun}, repeats=${repeats}`);
console.log("version,median_ms_per_row,all_ms_per_row");
for (const version of ["v1", "v2"]) {
  const input = request(version);
  const check = JSON.parse(session_json(input));
  if (check.error || check.step.token < 0 || check.step.token >= vocabulary) {
    throw new Error("sampling bridge returned an invalid response");
  }
  batch(input); // warmup
  const samples = [];
  for (let repeat = 0; repeat < repeats; repeat++) {
    const start = performance.now();
    batch(input);
    samples.push((performance.now() - start) / rowsPerRun);
  }
  samples.sort((a, b) => a - b);
  console.log(`${version},${samples[2].toFixed(3)},${samples.map((x) => x.toFixed(3)).join("|")}`);
}
