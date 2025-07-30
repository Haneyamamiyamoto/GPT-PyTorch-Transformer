# GPT-PyTorch-Transformer
An improved, from-scratch GPT-style language model implemented in PyTorch with best-practice enhancements for training and generation.
🚀 Project Overview

This repository contains a modular PyTorch implementation of a GPT-like transformer language model, featuring:

Byte-Pair Encoding tokenization using HuggingFace GPT2TokenizerFast

Pre-LayerNorm transformer blocks with GELU activations

Weight tying between token embeddings and the output projection

Mixed-precision training via torch.cuda.amp

Linear warm-up + cosine decay learning rate scheduler

Gradient clipping (Max‑norm) for stable optimization

Top‑k / Top‑p sampling and temperature control for text generation

Configuration via command-line arguments for easy experimentation

📦 Installation

Clone the repository:

Install dependencies:

pip install -r requirements.txt

Download or prepare your text corpus and place it in the data/ directory.

🏗️ Training

python train.py \
  --train_file data/wizard_of_oz.txt \
  --batch_size 32 \
  --block_size 128 \
  --max_iters 3000 \
  --lr 3e-4 \
  --warmup_steps 200 \
  --eval_interval 50

train_file: Path to your training text.

batch_size: Number of sequences per batch.

block_size: Context length for model inputs.

max_iters: Total training steps.

lr: Initial learning rate.

warmup_steps: Steps to linearly ramp up the learning rate.

eval_interval: How often (in steps) to evaluate on validation data.

📝 Generation

Once training completes, run:

python generate.py \
  --checkpoint checkpoints/best.pt \
  --prompt "Hello, world!" \
  --max_new_tokens 100 \
  --temperature 0.8 \
  --top_k 50 \
  --top_p 0.9

This produces a sampled continuation from the trained model.

⭐ Future Work

Integrate rotary positional embeddings (RoPE) or ALiBi for better long-context generalization

Swap in FlashAttention for faster, memory-efficient attention

Add distributed multi-GPU training support

Extend to byte-level tokenization and larger corpora

Developed by Girish Samdarshi | AI/ML Engineer
