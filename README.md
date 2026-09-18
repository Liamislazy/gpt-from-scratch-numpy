# gpt-from-scratch-numpy
This project is inspired by Green Code's 'Coding ChatGPT from Scratch' series and a desire to fully understand the architecture of an LLM like ChatGPT by building it via a pure NumPy implementation. Coded entirely from scratch without AI code generation to maximize first-principles learning.

## Project Structure

This is the structure of this project:

```text
gpt2_numpy/
├── src/
│   ├── __init__.py
│   ├── layers.py        # Custom Linear, LayerNorm, Embedding, activations
│   ├── model.py         # Multi-Head Attention, Transformer Block, GPT2 assembly
│   ├── optimizer.py     # AdamW optimizer & parameter updates in NumPy
│   └── utils.py         # BPE tokenizer wrapper, dataloader, batching
├── train.py             # Dedicated script for running training loops
├── generate.py          # Script for inference and text generation
├── explore.ipynb        # Notebook for sanity checks & visual debugging
└── requirements.txt
```