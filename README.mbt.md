# moonbit-mirostat

A model-independent MoonBit implementation of Mirostat adaptive sampling. A caller
passes one logit row and one uniform random draw per generated token. The library
returns the chosen token, its probability and surprise under the original model
distribution, the retained candidate count, and the updated feedback state.

The package implements both algorithms in the [Mirostat paper](https://arxiv.org/abs/2007.14966):
V1 estimates a Zipf exponent and chooses an adaptive top-k; V2 retains tokens
whose model surprise is at most the current `mu`. The feedback rule is
`mu := mu - eta * (observed_surprise - tau)`. `tau`, `mu`, and surprise use bits.

## When to use it

- A MoonBit text-generation loop can control the observed surprise of generated
  tokens without fixing a top-k or top-p threshold for every step.
- A WebAssembly application can run sampling locally after a model inference
  backend supplies logits, without routing token selection through JavaScript.
- A decoding experiment can replay identical logits and uniform draws through
  V1 and V2, then compare their retained vocabulary and surprise traces.

This package does not run a language model, tokenize text, fetch weights, or own
an RNG. The caller chooses how to obtain logits and randomness. It does not
claim exact behavioral equivalence to a particular model runtime: runtimes may
apply penalties, grammar masks, temperature, or other transformations before
Mirostat, and those transformations change the input distribution.

## Build and run

```sh
moon check --deny-warn
moon test --deny-warn
moon run cmd/main
moon run cmd/experiment
moon run cmd/generate
moon run cmd/batch
moon run cmd/weights
moon run cmd/process
moon run cmd/sweep
moon run cmd/json
```

`cmd/main` prints five reproducible decisions for each variant.
`cmd/experiment` runs both variants for 1000 draws from the same static Zipf
distribution and seed, then prints CSV summary metrics. This is a deterministic
algorithm fixture, not a language-model quality benchmark.
`cmd/generate` demonstrates a context-dependent callback and an end token
using a tiny deterministic model fixture.
`cmd/batch`, `cmd/weights`, and `cmd/process` demonstrate atomic batching,
weight inputs, and explicit logit preprocessing. `cmd/sweep` evaluates a set
of surprise targets on the same controlled fixture. `cmd/json` prints the JSON
wire formats for a WebAssembly host.

A typical integration creates one sampler per generation session:

```moonbit nocheck
let config = @mirostat.Config::new(3.0, eta=0.1).unwrap()
let sampler = @mirostat.Sampler::new(config, version=@mirostat.Version::v2())
let step = sampler.sample([4.0, 2.0, 1.0], 0.42).unwrap()
println(step.token())
```

The uniform draw must be finite and in `[0, 1)`. A model row must contain at
least one finite logit. Negative infinity masks a token; NaN and positive
infinity are errors. Token indices are zero based and preserve the incoming
logit order. Equal probabilities rank by smaller token index, so a fixed input
and draw produce a repeatable result. `reset()` restores initial `mu` and the
step count. `replay()` accepts aligned arrays of rows and draws, and
`summarize()` reports mean surprise, retained count, and absolute error against
the target.

If the model backend produces non-negative weights, use `sample_weights()`;
the package normalizes them. `preview()` and `preview_weights()` show the
candidate count, retained model mass, and expected surprise without drawing or
changing state. `ParkMiller` supplies a reproducible portable RNG for examples;
`sample_with_rng()` consumes it automatically. Production hosts can continue to
pass uniform draws from their own RNG.

`checkpoint()` and `restore()` support a generation branch that may need to
roll back; `fork()` creates a separate session with the same feedback state.
`replay_atomic()` restores state if a later row in a batch fails, while
`replay()` retains earlier successful steps. Neither function owns the caller's
random source.

`generate()` drives a context-dependent model callback with generated token
history and an optional end token. For a fixed distribution used repeatedly in
an experiment, `PreparedRow` caches its softmax and full ranking, and
`sample_prepared()` reuses them. `TraceAccumulator` computes online summary
metrics without retaining the whole trace.

For a model server that decodes several requests in one inference batch,
`sample_batch()` accepts aligned sampler, logit-row, and uniform arrays.
`sample_batch_atomic()` restores every sampler if any row fails. Each array
entry should refer to a distinct generation session.

## Host integration

For a host that makes the categorical draw itself, call `candidates()` to get
the token ids, original probabilities, and normalized candidate probabilities.
After selecting one of those token ids, call `observe_token()` to validate it
against the current Mirostat candidate set and update feedback. This is useful
when a model runtime already owns its random generator or implements a fused
sampling kernel.

The `jsonio` package provides a compact JSON protocol over the same public
library. `init_json` creates the initial portable state, `session_json` takes
one next-token row together with prior `state_mu` and `state_steps`,
`batch_json` processes multiple independent sessions atomically, and
`preview_json` inspects a candidate set without consuming a draw. `weight_json`
is the one-shot counterpart for a backend that returns non-negative weights.
These helpers do not perform model inference or file/network I/O.

## Validation scope

The test suite covers stable softmax, masked logits, normalization, candidate
selection, V1 and V2 feedback, state rollback, batch atomicity, JSON decoding,
fixed-seed replay, synthetic Zipf drift, and WebAssembly-compatible execution.
The synthetic experiments verify that the feedback loop approaches feasible
target surprise levels for their fixtures. They do not establish text quality,
model quality, or performance for a particular vocabulary size.

## Numerical and state behavior

Softmax subtracts the largest logit before exponentiation. The Zipf slope
estimator uses adjacent ranked probabilities up to `m`; a flat or underflowed
distribution uses a small positive limiting slope. V1 clamps estimated `k` to
the vocabulary. V2 always retains the most probable token, even if its
surprise exceeds `mu`. The selected token's surprise is measured against the
original softmax, before truncation and renormalization. An invalid row or draw
returns an error without changing sampler state. In a replay, earlier valid
rows remain committed if a later row fails.

V1 uses a bounded heap to inspect the highest `m` probabilities and then
select its adaptive top-k candidates. V2 scans the vocabulary and sorts only
the tokens below its surprise cutoff. Both methods still inspect every logit
at each step. Large-vocabulary low-latency decoders should benchmark the
package with their model and target settings before adoption.

For controlled comparisons, `sample_top_k`, `sample_top_p`, and
`sample_temperature` use the same stable probability and draw routines.
Additional comparison samplers cover min-p, epsilon, and locally typical
sampling. A caller may apply `mask_logits`, `bias_logits`, or
`penalize_repetition` before Mirostat; these transformations are explicit so
the reported surprise is based on the transformed model row.
`entropy`, `prefix_mass`, `expected_prefix_surprise`, and `perplexity` provide
small diagnostics for comparing distributions with an observed trace.

## Origin and license

This is an original MoonBit implementation of the algorithms and equations in
Sourya Basu et al., *Mirostat: A Neural Text Decoding Algorithm That Directly
Controls Perplexity*, ICLR 2021. The [authors' Python research code](https://github.com/basusourya/mirostat)
was used as a reference for terminology and algorithm scope; this repository
does not copy its source files. The upstream code is MIT licensed. This
MoonBit package is licensed under Apache-2.0; see [LICENSE](LICENSE).
