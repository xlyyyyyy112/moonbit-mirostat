# Hugging Face integration validation

Date: 2026-10-21

## Test setup

- Host: Windows 11, CPU-only PyTorch 2.10.0, two Torch threads
- Transformers: 4.57.1
- Model: `sshleifer/tiny-gpt2`, revision `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`
- Weight file: 2,514,146 bytes; SHA-256 `b706b24034032bdfe765ded5ab6403d201d295a995b790cb24c74becca5c04e6`
- Prompt: `The capital of France is`
- Seed: 7; 16 generated tokens per run

The model files and JSON reports are stored under `E:\moonbit-mirostat`, outside this repository. The run used the plain-text tokenizer path because this model has no chat template.

## Results

| Sampler | Target | Tokens | Generation tokens/s | Bridge ms/token | Mean observed surprise | Kept candidates | Output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Mirostat v1 | 4 bits | 16 | 9.20 | 98.73 | 15.392 bits | 1 | `stairs` repeated 16 times |
| Mirostat v2 | 4 bits | 16 | 9.82 | 92.60 | 15.392 bits | 1 | `stairs` repeated 16 times |
| Top-p | 0.9 | 16 | 41.54 | n/a | n/a | n/a | Varied tokens, incoherent text |

Both Mirostat versions completed the Python -> Node -> MoonBit sampling loop, advanced session state, returned in-range token IDs, and produced a per-token trace. The top-p run used the same model, prompt, seed, and token count.

The default 4-bit target is below this tiny model's observed token surprise. The run therefore retained one candidate and repeated it. This confirms that integration works; it does not establish useful text quality. A separate v2 diagnostic at `tau=15` retained the full 50,257-token vocabulary and produced varied, still incoherent text. That setting is diagnostic only, not a recommended default.

Bridge time was about 92.6-98.7 ms per token for this 50k-token vocabulary. This is a material cost for a tiny model and needs measurement and optimization before making performance claims on larger vocabularies.

## Limits and outstanding work

`tiny-gpt2` is a minimal integration fixture, not a representative language model. These results do not validate natural-language quality, target-surprise tracking on a usable model, or throughput for Qwen.

The planned `Qwen/Qwen2.5-0.5B-Instruct` weight file is not complete. The Hugging Face CDN repeatedly stalled or timed out on this network. A `TinyStories-8M` attempt also stalled after a few megabytes and was stopped; neither incomplete download was loaded. Full-model quality and performance comparison remain unverified.

The host runner now supplies an explicit attention mask and reads UTF-8 JSON metadata with or without a Windows BOM. The Qwen download, followed by matched-seed Mirostat v1/v2 and top-p runs, remains the next validation step.
