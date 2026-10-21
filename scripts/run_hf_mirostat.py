"""Generate text with a Hugging Face causal LM and MoonBit Mirostat sampling."""

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(r"E:\moonbit-mirostat\models\Qwen2.5-0.5B-Instruct")
BRIDGE = ROOT / "integrations" / "hf_mirostat" / "bridge.cjs"


class MoonBitSampler:
    def __init__(self, node: str, version: str, tau: float, eta: float, seed: int):
        self.process = subprocess.Popen(
            [node, str(BRIDGE)],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.version = version
        self.tau = tau
        self.eta = eta
        self.mu = 2.0 * tau
        self.steps = 0
        self.rng = random.Random(seed)
        self.bridge_seconds = 0.0
        self.sampling_host_seconds = 0.0
        self.trace: list[dict] = []

    def choose(self, logits: list[float]) -> int:
        request = {
            "logits": logits,
            "tau": self.tau,
            "eta": self.eta,
            "uniform": self.rng.random(),
            "version": self.version,
            "state_mu": self.mu,
            "state_steps": self.steps,
        }
        start = time.perf_counter()
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(request, separators=(",", ":"), allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        self.bridge_seconds += time.perf_counter() - start
        if not line:
            raise RuntimeError("MoonBit bridge exited without a response")
        response = json.loads(line)
        if response.get("error"):
            raise RuntimeError(f"MoonBit sampling rejected model logits: {response}")
        token = int(response["step"]["token"])
        if not 0 <= token < len(logits):
            raise RuntimeError(f"MoonBit returned out-of-range token id {token}")
        self.mu = float(response["state_mu"])
        self.steps = int(response["state_steps"])
        step = response["step"]
        self.trace.append(
            {
                "token_id": token,
                "surprise_bits": float(step["surprise_bits"]),
                "kept_tokens": int(step["kept_tokens"]),
                "mu_after": self.mu,
            }
        )
        return token

    def close(self) -> None:
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        code = self.process.wait(timeout=10)
        if code != 0:
            raise RuntimeError(f"MoonBit bridge exited with status {code}")


class MirostatLogitsProcessor(LogitsProcessor):
    def __init__(self, sampler: MoonBitSampler, temperature: float):
        self.sampler = sampler
        self.temperature = temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.ndim != 2 or scores.shape[0] != 1:
            raise ValueError("the integration currently supports batch size 1")
        start = time.perf_counter()
        logits = (scores[0] / self.temperature).detach().to(device="cpu", dtype=torch.float32).tolist()
        if not all(math.isfinite(value) for value in logits):
            raise ValueError("the model produced non-finite logits; masked logits are not supported by this JSON bridge")
        token = self.sampler.choose(logits)
        self.sampler.sampling_host_seconds += time.perf_counter() - start
        scores.fill_(float("-inf"))
        scores[0, token] = 0.0
        return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--sampler", choices=("mirostat", "top-p"), default="mirostat")
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--node", default="node")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    numeric_settings = (args.tau, args.eta, args.temperature, args.top_p)
    if (
        args.max_new_tokens < 1
        or args.threads < 1
        or not all(math.isfinite(value) and value > 0 for value in numeric_settings[:3])
        or not 0 < args.top_p <= 1
    ):
        parser.error("max-new-tokens, threads, tau, eta and temperature must be positive; top-p must be in (0, 1]")
    if not (args.model / "config.json").is_file():
        parser.error(f"model files not found under {args.model}; run scripts/download_hf_model.py first")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": args.prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    sampler = (
        MoonBitSampler(args.node, args.version, args.tau, args.eta, args.seed)
        if args.sampler == "mirostat"
        else None
    )
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature if sampler is None else 1.0,
                top_k=0,
                top_p=args.top_p if sampler is None else 1.0,
                repetition_penalty=1.0,
                logits_processor=(
                    [MirostatLogitsProcessor(sampler, args.temperature)]
                    if sampler is not None
                    else None
                ),
                use_cache=True,
            )
    finally:
        elapsed = time.perf_counter() - start
        if sampler is not None:
            sampler.close()

    generated_ids = output[0, input_ids.shape[-1] :].tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    trace = sampler.trace if sampler is not None else []
    mean_surprise = (
        sum(step["surprise_bits"] for step in trace) / len(trace)
        if trace
        else None
    )
    download_record = args.model / "download.json"
    revision = (
        json.loads(download_record.read_text(encoding="utf-8-sig")).get("revision")
        if download_record.exists()
        else None
    )
    report = {
        "model": str(args.model.resolve()),
        "model_revision": revision,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "version": args.version,
        "sampler": args.sampler,
        "tau_bits": args.tau,
        "eta": args.eta,
        "temperature": args.temperature,
        "top_p": args.top_p if sampler is None else None,
        "seed": args.seed,
        "threads": args.threads,
        "prompt_tokens": int(input_ids.shape[-1]),
        "generated_tokens": len(generated_ids),
        "generated_token_ids": generated_ids,
        "total_seconds": elapsed,
        "tokens_per_second": len(generated_ids) / elapsed if elapsed else 0.0,
        "bridge_seconds": sampler.bridge_seconds if sampler is not None else 0.0,
        "sampling_host_seconds": (
            sampler.sampling_host_seconds if sampler is not None else 0.0
        ),
        "bridge_ms_per_token": (
            1000 * sampler.bridge_seconds / max(1, sampler.steps)
            if sampler is not None
            else 0.0
        ),
        "mean_surprise_bits": mean_surprise,
        "final_mu": sampler.mu if sampler is not None else None,
        "trace": trace,
        "text": text,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
