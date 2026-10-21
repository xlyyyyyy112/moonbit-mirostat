"""Download a pinned public model revision outside the repository."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEFAULT_DESTINATION = Path(r"E:\moonbit-mirostat\models\Qwen2.5-0.5B-Instruct")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()

    revision = args.revision
    HfApi().model_info(args.model, revision=revision)
    args.destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model,
        revision=revision,
        local_dir=str(args.destination),
    )
    metadata = {
        "model": args.model,
        "revision": revision,
        "path": str(args.destination.resolve()),
    }
    (args.destination / "download.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
