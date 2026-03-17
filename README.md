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
  <a href=""><b>Paper Link (Coming Soon)</b>👁️</a>
</div>

## Table of Contents

1. [Introduction](#1-introduction)
2. [Datasets](#2-datasets)
3. [Model Architecture](#3-model-architecture)
4. [Model Downloads](#4-model-downloads)
5. [How to Run Locally](#5-how-to-run-locally)
6. [Evaluation Results](#6-evaluation-results)
7. [Contributions](#7-contributions)
8. [Citation](#8-citation)
9. [Contact](#9-contact)

## 1. Introduction
We present Mochi (**M**alicious **O**utput **C**uration for **H**igh-quality **I**njection-defense), an end-to-end system with the goal of training small language models to classify and understand prompt injection. Our work begins with two curated datasets (classification and cleaning) built on top of [StrongREJECT](https://arxiv.org/pdf/2402.10260), [AdvBench](https://arxiv.org/pdf/2307.15043), [Berkeley SafeGuard](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection), and [Databricks Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k), and enhanced with Claude Sonnet models. We fine-tune the following small language models on our classification dataset using Low Rank Approximation (LoRA) to classify between benign and malicious models: [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct), [Qwen-3-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), and [Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B). Furthermore, we fine-tune the Llama model further on our cleaning dataset to demonstrate that small language models can build contextual awareness about prompts. Mochi models achieve state-of-the-art accuracy, precision, and recall scores on classification objectives. They also perform strongly on cleaning and cross-task objectives, which we highlight in [Evaluation Results](#6-evaluation-results). 

## 2. Datasets

## 3. Model Architecture

## 4. Model Downloads

## 5. How to Run Locally

## 6. Evaluation Results

## 7. Contributions

## 8. Citation

## 9. Contact
If you have any questions, please raise an issue on the repository.

