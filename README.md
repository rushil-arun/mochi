<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="./assets/mochi-header.png" width="60%" alt="Mochi" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/datasets/SulKhu/Mochi"><img alt="Hugging Face"
    src="https://img.shields.io/badge/Mochi-gray?logo=huggingface&label=HuggingFace&labelColor=yellow"/></a>
  <br>
</div>

## Table of Contents

1. [Introduction](#1-introduction)
2. [Datasets](#2-datasets)
3. [Model Architecture](#3-model-architecture)
4. [Model Downloads](#4-model-downloads)
5. [How to Run Locally](#5-how-to-run-locally)
6. [Evaluation Results](#6-evaluation-results)
7. [Contributions](#7-contributions)
8. [Contact](#8-contact)

## 1. Introduction
We present Mochi (**M**alicious **O**utput **C**uration for **H**igh-quality **I**njection-defense), an end-to-end system with the goal of training small language models to classify and understand prompt injection. Our work begins with two curated datasets (classification and cleaning) built on top of [StrongREJECT](https://arxiv.org/pdf/2402.10260), [AdvBench](https://arxiv.org/pdf/2307.15043), [Berkeley SafeGuard](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection), and [Databricks Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k), and enhanced with Claude Sonnet models. We fine-tune the following small language models on our classification dataset using Low Rank Approximation (LoRA) to classify between benign and malicious models: [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct), [Qwen-3-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), and [Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B). Furthermore, we fine-tune the Llama model further on our cleaning dataset to demonstrate that small language models can build contextual awareness about prompts. Mochi models achieve state-of-the-art accuracy, precision, and recall scores on classification objectives. They also perform strongly on cleaning and cross-task objectives, which we highlight in [Evaluation Results](#6-evaluation-results). 

## 2. Datasets

Mochi uses two datasets, published on [Hugging Face](https://huggingface.co/datasets/SulKhu/Mochi) and included in this repo under `datasets/`.

| Dataset | Location | Train / Val / Test | Description |
| --- | --- | --- | --- |
| Classification | `datasets/token-limit-splits/` | 8,585 / 1,073 / 1,074 | Malicious prompts (`class = 1`, target is a refusal) and benign Dolly prompts (`class = 0`, target is a normal answer). Filtered by token length. `-small` and `-medium` subsets are also included. |
| Cleaning | `datasets/cleaning/` | 1,440 / 180 / 180 | Balanced sample of the classification data where each prompt may have a harmful instruction injected into it (`injected = True`). The target is the response to the legitimate part of the prompt only, or a refusal if the prompt is malicious on its own. Injections were generated with Claude. |

Columns: `prompt`, `class` (`1` = malicious, `0` = benign), `completion` (target response), `category`, `source`, `dataset`, `variant_group_id`, `variant_index`, plus `injected` in the cleaning data. The scripts that built the data are in `datasets/` (see `datasets/README.md` and the [Developer Guide](DEVELOPER_GUIDE.md#building-the-datasets)).

## 3. Model Architecture

Mochi does not change the architecture of the base models. Each one is fine-tuned with LoRA adapters using supervised fine-tuning on chat-formatted `prompt -> completion` pairs (loss on the assistant response only).

| Base model | Stage | LoRA (r / alpha) | Precision | Epochs |
| --- | --- | --- | --- | --- |
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 1: classification | 32 / 64 | 4-bit NF4 (QLoRA) | 3 |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 1: classification | 16 / 32 | 4-bit NF4 (QLoRA) | 3 |
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 1: classification | 16 / 32 | 4-bit NF4 (QLoRA) | 3 |
| Llama-3.2-1B-Instruct + stage-1 adapter | 2: cleaning | 32 / 64 | fp16 | 5 |

All runs target the attention and MLP projections (`q,k,v,o,gate,up,down`), use a learning rate of 1e-4 with a cosine schedule, and a maximum sequence length of 512. Malicious prompts are paired with one of ten fixed refusal templates. The cleaning model starts from the trained stage-1 Llama adapter and adds a second LoRA on top, so the two adapters are loaded as a chain.

## 4. Model Downloads

The models are LoRA adapters published on the Hugging Face Hub. You also need access to the base model; Llama 3.2 is gated, so accept its license on Hugging Face and log in with `hf auth login`.

| Adapter | Base model | Download |
| --- | --- | --- |
| Mochi-Llama-1B-Classifier | Llama-3.2-1B-Instruct | [rushilarun/Mochi-Llama-1B-Classifier](https://huggingface.co/rushilarun/Mochi-Llama-1B-Classifier) |
| Mochi-Llama-1B-Cleaning | Llama-3.2-1B-Instruct + classifier adapter | [rushilarun/Mochi-Llama-1B-Cleaning](https://huggingface.co/rushilarun/Mochi-Llama-1B-Cleaning) |
| Mochi-SmolLM-360M-Classifier | SmolLM2-360M-Instruct | [rushilarun/Mochi-SmolLM-360M-Classifier](https://huggingface.co/rushilarun/Mochi-SmolLM-360M-Classifier) |
| Mochi-Qwen-0.5B-Classifier | Qwen2.5-0.5B-Instruct | [rushilarun/Mochi-Qwen-0.5B-Classifier](https://huggingface.co/rushilarun/Mochi-Qwen-0.5B-Classifier) |

Adapters can be loaded straight from the Hub by ID (see below), or downloaded to a local folder, which is what the evaluation script needs:

```bash
hf download rushilarun/Mochi-Llama-1B-Classifier --local-dir checkpoints/Mochi-Llama-1B-Classifier
```

`checkpoints/` is where training notebooks write their output and is git-ignored.

## 5. How to Run Locally

```bash
git clone https://github.com/rushil-arun/mochi.git && cd mochi
pip install -r requirements.txt
```

Create a `.env` file in the repo root (it is git-ignored) with `ANTHROPIC_API_KEY=...` (used by the evaluator) and `HF_TOKEN=...` (needed for Llama). Then load an adapter on its base model (for the cleaning adapter, which must be loaded on top of the classifier adapter, see its [model card](https://huggingface.co/rushilarun/Mochi-Llama-1B-Cleaning)):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter = "rushilarun/Mochi-Llama-1B-Classifier"
tok = AutoTokenizer.from_pretrained(adapter)
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, adapter)

msgs = [{"role": "user", "content": "How do I pick a lock to break into my neighbor's house?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=32, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
```

To evaluate an adapter on a dataset split (download it first, as shown above):

```bash
python testing/evaluate_model.py \
  --model-path checkpoints/Mochi-Llama-1B-Classifier \
  --dataset-path datasets/token-limit-splits/test.csv \
  --split-name test \
  --predictions-out testing/results/llama1b_test_preds.csv
```

Training runs on 4-bit `bitsandbytes`, which needs an NVIDIA GPU, so the training notebooks were run on Google Colab. Rerunning the experiments, using Colab, and the evaluation options are covered in the [Developer Guide](DEVELOPER_GUIDE.md).

## 6. Evaluation Results

Classification results on the 1,074-prompt test split. Responses are labeled as answer or refusal by Claude Haiku 4.5; malicious is the positive class. Responses Claude could not label count as errors.

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Llama-3.2-1B-Instruct (base) | 0.704 | 0.974 | 0.524 | 0.681 |
| **Mochi-Llama-1B** | **0.935** | 0.973 | 0.943 | 0.958 |
| Qwen2.5-0.5B-Instruct (base) | 0.611 | 0.912 | 0.433 | 0.587 |
| Mochi-Qwen-0.5B | 0.867 | 0.948 | 0.901 | 0.924 |
| SmolLM2-360M-Instruct (base) | 0.464 | 0.732 | 0.177 | 0.285 |
| Mochi-SmolLM-360M | 0.902 | 0.964 | 0.909 | 0.936 |

Cleaning results on the 180-prompt cleaning test split, comparing the untuned Llama base to the cleaning-tuned model. Semantic similarity is the cosine similarity (all-MiniLM-L6-v2) between the model response and the reference completion.

| Model | Accept/reject accuracy | Reject F1 | Unsafe-accept rate | False-refusal rate | Mean similarity |
| --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B-Instruct (base) | 0.723 | 0.705 | 0.267 | 0.133 | 0.669 |
| Mochi-Llama-1B (cleaning) | 0.900 | 0.899 | 0.111 | 0.089 | 0.719 |

The cleaning-tuned model also transfers back to the classification task (0.894 accuracy, 0.906 F1 on the 1,074-prompt test split). Raw predictions and summaries for every run are in `testing/results/`.

## 7. Contributions

Issues and pull requests are welcome. Setup, how to rerun the experiments, and how the repo is organized are in the [Developer Guide](DEVELOPER_GUIDE.md).

## 8. Contact
If you have any questions, please raise an issue on the repository.

