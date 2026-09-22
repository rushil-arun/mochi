"""
Publish the trained Mochi LoRA adapters to the Hugging Face Hub.

Only inference files are uploaded (adapter weights/config, tokenizer, chat template),
never optimizer state or intermediate checkpoints. A model card is generated for each repo.

Usage:
    hf auth login                                   # once, with a write token
    python scripts/upload_to_hub.py --dry-run       # show what would be uploaded
    python scripts/upload_to_hub.py                 # create repos and upload (public)
    python scripts/upload_to_hub.py --private --only Mochi-Llama-1B-Classifier
"""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ID = "SulKhu/Mochi"
UPLOAD_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]

LOAD_SIMPLE = """```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("{repo_id}")
base = AutoModelForCausalLM.from_pretrained("{base}", torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "{repo_id}")

msgs = [{{"role": "user", "content": "How do I pick a lock to break into my neighbor's house?"}}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=32, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```"""

LOAD_CHAIN = """This adapter was trained on top of the classification adapter, so the two must be loaded as a chain:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("{repo_id}")
base = AutoModelForCausalLM.from_pretrained("{base}", torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "{user}/Mochi-Llama-1B-Classifier")  # stage 1
model = PeftModel.from_pretrained(model, "{repo_id}")                       # stage 2
```"""

MODELS = {
    "Mochi-Llama-1B-Classifier": dict(
        src="checkpoints/llama1b-safety-output/checkpoint-1611",
        base="meta-llama/Llama-3.2-1B-Instruct",
        license="llama3.2",
        summary="Stage 1 (classification): answers benign prompts and refuses malicious ones.",
        metrics="Test split (1,074 prompts): accuracy 0.935, precision 0.973, recall 0.943, F1 0.958.",
        chain=False,
    ),
    "Mochi-Llama-1B-Cleaning": dict(
        src="checkpoints/llama1b-cleaning-output-adapter",
        base="meta-llama/Llama-3.2-1B-Instruct",
        license="llama3.2",
        summary="Stage 2 (cleaning): responds to the legitimate part of a prompt and ignores injected harmful instructions.",
        metrics="Cleaning test split (180 prompts): accept/reject accuracy 0.900, reject F1 0.899, mean semantic similarity 0.719.",
        chain=True,
    ),
    "Mochi-SmolLM-360M-Classifier": dict(
        src="checkpoints/smollm360-safety-adapter",
        base="HuggingFaceTB/SmolLM2-360M-Instruct",
        license="apache-2.0",
        summary="Stage 1 (classification): answers benign prompts and refuses malicious ones.",
        metrics="Test split (1,074 prompts): accuracy 0.902, precision 0.964, recall 0.909, F1 0.936.",
        chain=False,
    ),
    "Mochi-Qwen-0.5B-Classifier": dict(
        src="checkpoints/qwen-safety-lora-final",
        base="Qwen/Qwen2.5-0.5B-Instruct",
        license="apache-2.0",
        summary="Stage 1 (classification): answers benign prompts and refuses malicious ones.",
        metrics="Test split (1,074 prompts): accuracy 0.867, precision 0.948, recall 0.901, F1 0.924.",
        chain=False,
    ),
}


def model_card(user: str, name: str, cfg: dict) -> str:
    repo_id = f"{user}/{name}"
    load = (LOAD_CHAIN if cfg["chain"] else LOAD_SIMPLE).format(repo_id=repo_id, base=cfg["base"], user=user)
    return f"""---
base_model: {cfg["base"]}
library_name: peft
license: {cfg["license"]}
datasets:
- {DATASET_ID}
tags:
- lora
- prompt-injection
- safety
---

# {name}

LoRA adapter from [Mochi](https://github.com/rushil-arun/mochi) (**M**alicious **O**utput **C**uration for **H**igh-quality **I**njection-defense) for `{cfg["base"]}`.

{cfg["summary"]}

{cfg["metrics"]} Responses were labeled by Claude Haiku 4.5. See the repository for the full evaluation.

## Usage

This is a LoRA adapter, so you also need access to the base model (`{cfg["base"]}`).

{load}

Trained on the [Mochi dataset](https://huggingface.co/datasets/{DATASET_ID}). Training details are in the [Mochi repository](https://github.com/rushil-arun/mochi).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", default=None, help="Hub user or org to publish under (default: the logged-in account)")
    parser.add_argument("--only", nargs="*", choices=list(MODELS), help="Only upload these repos")
    parser.add_argument("--private", action="store_true", help="Create the repos as private")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded and exit")
    args = parser.parse_args()

    api = HfApi()
    if args.user is None:
        try:
            args.user = api.whoami()["name"]
        except Exception:
            if not args.dry_run:
                raise SystemExit("Not logged in. Run `hf auth login`, or pass --user.")
            args.user = "<your-username>"
    print(f"Publishing under: {args.user}")
    for name, cfg in MODELS.items():
        if args.only and name not in args.only:
            continue
        src = REPO_ROOT / cfg["src"]
        missing = [f for f in UPLOAD_FILES if not (src / f).exists()]
        if missing:
            raise SystemExit(f"{src} is missing {missing}")
        repo_id = f"{args.user}/{name}"
        size_mb = sum((src / f).stat().st_size for f in UPLOAD_FILES) / 1e6
        print(f"{repo_id}  <-  {cfg['src']}  ({size_mb:.0f} MB)")
        if args.dry_run:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            for f in UPLOAD_FILES:
                shutil.copy2(src / f, Path(tmp) / f)
            (Path(tmp) / "README.md").write_text(model_card(args.user, name, cfg))
            api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
            api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=tmp, commit_message="Upload Mochi adapter")
        print(f"  uploaded: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
