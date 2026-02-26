# BitNet Fine-Tuning on IMDB

This repository fine-tunes a small pretrained transformer (`bert-tiny`) into a **BitNet-style model with ternary weights** for sentiment classification on the IMDB dataset. It replaces all linear layers with custom `BitLinear` layers that quantize weights to `{-1, 0, 1}` during forward passes, using straight-through estimator for backpropagation.

## Why BitNet?
- Extreme weight quantization (1.58-bit) reduces memory footprint.

## Quickstart (Colab)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/bitnet-imdb/blob/main/notebooks/bitnet_imdb_colab.ipynb)

Run the notebook to fine-tune the model in minutes (free GPU recommended).

## Local Setup
```bash
git clone https://github.com/yourusername/bitnet-imdb.git
cd bitnet-imdb
pip install -r requirements.txt
python train.py --max_samples 5000   # quick test
