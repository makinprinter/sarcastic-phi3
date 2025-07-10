# Sarcastic Phi-3 Chatbot

This repository contains code for training and serving a conversational AI model fine-tuned to generate sarcastic responses. It uses Microsoft's `phi-3-mini-128k-instruct` model, fine-tuned with Low-Rank Adaptation (LoRA) for parameter-efficient learning.

## Overview

The project includes:

- **Training Pipeline** (`train.py`): Fine-tunes the base model on a dataset of sarcastic instructions and responses using the TRL `SFTTrainer` and PEFT's LoRA integration.
- **Inference Script** (`main.py`): Runs an interactive chatbot in the terminal using the fine-tuned model.
- **LoRA Adapter**: Stored and reused from `./sarcastic-phi3-model` for efficient inference.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
