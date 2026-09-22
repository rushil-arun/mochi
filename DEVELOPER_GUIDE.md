# Developer Guide

Details for setting up, rerunning the experiments, and working with the code. For an overview of the project, see the [README](README.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `datasets/` | Data-building scripts plus the final splits. `token-limit-splits/` is the classification data and `cleaning/` is the injected-prompt data. See `datasets/README.md`. |
| `finetuning/` | Training notebooks (one per model/stage) and the injection dataset generator. |
| `testing/` | `evaluate_model.py` (CLI evaluator), evaluation notebooks, and `results/` (saved predictions and summaries). |
| `checkpoints/` | Trained LoRA adapters and Trainer checkpoints (git-ignored, too large to commit). |
| `models/` | Small Qwen adapters committed to git. |
| `playground/` | Scratch scripts for trying out SLMs. See `playground/README.md`. |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...        # only for the gated meta-llama/Llama-3.2-1B-Instruct model
```

The Hugging Face account must have accepted the Llama 3.2 license.

### Local vs. Colab

Training uses 4-bit quantization (`bitsandbytes`), which needs an NVIDIA GPU, so the notebooks were run on Google Colab. Every notebook starts with a setup cell that works in both places:

- **Locally:** open the notebook from inside the repo. The setup cell finds the repo root and reads keys from `.env` or your environment.
- **Colab:** put the repo folder in Google Drive at `MyDrive/mochi` (edit `PROJECT_ROOT` in the first cell if it lives elsewhere). Drive is mounted for you. Keys are read from `.env` in that folder or from Colab Secrets (`ANTHROPIC_API_KEY`, and `Llama1B` for the Hugging Face token).

All data and output paths in the notebooks are relative to `PROJECT_ROOT`.

## Using the pretrained adapters

The adapters are published on the Hugging Face Hub (see the README for links) and load on top of their base model with `peft`. The Hub copies contain only inference files; the original training output, including optimizer state and intermediate checkpoints, lives in `checkpoints/` and is not in git. The cleaning adapter was trained on top of the classification adapter, so it must be loaded as a chain (base model, then classification adapter, then cleaning adapter). `testing/run_similarity_evaluation_colab-finetuned-base.ipynb` does this using `ORIGINAL_BASE_MODEL` and `PARENT_ADAPTER_PATH`.

## Publishing the adapters

`scripts/upload_to_hub.py` uploads the final adapters (weights, config, tokenizer, chat template only) and generates a model card for each. It reads from `checkpoints/`, so run it from a copy of the project that has the trained weights.

```bash
hf auth login                              # once, with a write token
python scripts/upload_to_hub.py --dry-run  # list what would be uploaded
python scripts/upload_to_hub.py            # create the repos and upload
```

Options: `--user` (Hub user or org, default the account you are logged in as; you need write access to it), `--only <repo names>`, `--private`. The mapping from repo name to source checkpoint is the `MODELS` dict at the top of the script; update it if you retrain.

## Evaluating a model

`testing/evaluate_model.py` generates a response for every prompt in a CSV, has Claude label each response as an answer or a refusal, and reports accuracy, precision, recall, and F1 against the `class` column.

```bash
python testing/evaluate_model.py \
  --model-path checkpoints/Mochi-Llama-1B-Classifier \
  --dataset-path datasets/token-limit-splits/test.csv \
  --split-name test \
  --predictions-out testing/results/llama1b_test_preds.csv
```

Useful flags: `--base-model` (if it can't be read from `adapter_config.json`), `--limit N` for a quick run, `--generation-batch-size`, `--claude-batch-size`. Pass a Hub ID such as `meta-llama/Llama-3.2-1B-Instruct` as `--model-path` to evaluate an untuned base model. In the saved predictions, `claude_pred_label` is `1` for a refusal, `0` for an answer, and `-1` if Claude's label could not be parsed.

The evaluation notebooks do the same interactively. `run_evaluation_colab.ipynb` and `run_evaluation_colab-base.ipynb` cover classification. `run_similarity_evaluation_colab-finetuned-base.ipynb` evaluates the cleaning model with accept/reject metrics plus semantic similarity to the reference (`all-MiniLM-L6-v2`). Edit `MODEL_PATH` and `DATASET_PATH` at the top of each notebook to choose the checkpoint and split.

Results from the original runs are in `testing/results/`.

## Rerunning the experiments

Run the notebooks in `finetuning/` in this order. The data is already in `datasets/`, so you can start at step 1.

1. **Classification tuning.** `llama1B_safety_tuning.ipynb` (also `smollm360_safety_finetuning.ipynb` and `qwen_safety_qlora_colab.ipynb`). Trains a 4-bit QLoRA adapter on `datasets/token-limit-splits/{train,val}.csv`, evaluates on train and validation, and saves checkpoints to `checkpoints/<run>-output/` and the final adapter to `checkpoints/<run>-adapter`.
2. **Build the injection dataset (optional).** `injection_dataset_generator.ipynb` samples 500 malicious and 500 benign prompts from the classification train split, asks Claude to inject an attack string into each, and writes `datasets/cleaning/dataset.csv` and its splits. Needs `ANTHROPIC_API_KEY`. The result is already in `datasets/cleaning/`.
3. **Cleaning tuning.** `llama1b_cleaning_tuning.ipynb` loads the classification adapter (`CHECKPOINT_DIR`, default `checkpoints/llama1b-safety-output/checkpoint-1611`) and continues training on `datasets/cleaning/`. Checkpoints go to `checkpoints/llama1b-cleaning-output/`.
4. **Evaluate** as described above.

Each run overwrites its output directory under `checkpoints/`, so copy existing weights elsewhere first if you want to keep them. Injection generation and evaluation call the Claude API and cost money.

## Building the datasets

The final splits are committed, but the scripts in `datasets/` show how they were produced: combine a malicious-prompt dataset with Dolly benign prompts (`build_combined_safety_dataset.py`), generate grammatical variants with Claude (`generate_dataset.py`, `generate_grammatical_variants.py`, `add_context.py`), filter by token length (`plot_token_lengths.py`, `cut_token_lengths.py`), and split (`split_dataset.py`). See `datasets/README.md`.

## Data columns

| Column | Meaning |
| --- | --- |
| `prompt` | Input text given to the model |
| `class` (`label` in some notebooks) | `1` = malicious (should be refused), `0` = benign |
| `completion` | Target response: the benign answer, or a refusal template for malicious prompts |
| `injected` | Cleaning data only: whether an injection string was added to the prompt |
| `category`, `source`, `dataset`, `variant_group_id`, `variant_index` | Provenance metadata |
