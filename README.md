# TurboQuant-GPT-2-Implementation

A from-scratch PyTorch implementation of GPT-2 (124M), built as the substrate for applying **TurboQuant**, an extreme-compression KV cache quantization scheme from Google Research (arXiv:2504.19874). The base model reproduces the architecture and training recipe of the original GPT-2, trained on the FineWeb-Edu corpus and evaluated on HellaSwag. The TurboQuant inference pipeline is layered on top.

> **Status:** Work in progress. The base GPT-2 implementation and training loop are complete. The KV cache, randomized rotation, Lloyd-Max quantization, and QJL 1-bit pipeline are under active development.

---

## Motivation

KV cache memory is the dominant cost of long-context LLM inference. Storing the cached keys and values in FP16 grows linearly with sequence length and quickly exceeds the activation footprint of the model itself. TurboQuant compresses this cache to as low as 1–2 bits per element with negligible quality loss, by combining three ideas:

1. A randomized orthogonal rotation that Gaussianizes the per-channel distribution of cached tensors, dramatically reducing the dynamic range that scalar quantizers must cover.
2. A Lloyd-Max scalar codebook fit to the rotated distribution, replacing uniform quantization for a strict accuracy gain at the same bit-width.
3. A quantized Johnson-Lindenstrauss transform (QJL) for the 1-bit regime, where sign-bit encoding is debiased through an asymmetric inner-product estimator.

This repository implements the full pipeline against a self-contained GPT-2 model rather than treating it as a black-box library, so every tensor of interest is observable and modifiable.

## Repository contents

```
my_gpt2.py        Full GPT-2 model + training loop + HellaSwag eval + generation
fineweb.py        FineWeb-Edu shard download and tokenization
hellaswag.py      HellaSwag dataset loader and rendering for zero-shot eval
requirements.txt  Python dependencies
```

## Architecture

The model follows the standard GPT-2 pre-norm decoder stack:

| Component | Specification |
| --- | --- |
| Layers | 12 |
| Heads | 12 |
| Embedding dim | 768 |
| Head dim | 64 |
| Context length | 1024 |
| Vocab size | 50304 (padded from 50257 for kernel efficiency) |
| Activation | GELU (exact, not approximate) |
| Attention | Flash Attention via `F.scaled_dot_product_attention` |
| Weight tying | Token embeddings shared with LM head |

Training uses bfloat16 autocast on CUDA, TF32 matmul precision, gradient accumulation to a total batch of 2^19 tokens, AdamW with `betas=(0.9, 0.95)` and `weight_decay=0.1`, gradient norm clipping at 1.0, and a cosine learning-rate schedule with a 715-step linear warmup decaying from `6e-4` to `6e-5` over 19,073 steps.

## Dataset

Pretraining uses [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), the educational subset of FineWeb. The `fineweb.py` script shards the corpus into pre-tokenized `.npy` files of approximately 100M tokens each, stored under `edu_fineweb10B/`. The 10B-token sample is sufficient to reach Karpathy's published baseline on HellaSwag for GPT-2 124M.

## Setup

```bash
git clone https://github.com/meltyblend/TurboQuant-GPT-2-Implementation.git
cd TurboQuant-GPT-2-Implementation
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tokenize the dataset (one-time, several hours):

```bash
python fineweb.py
```

## Training

```bash
python my_gpt2.py
```

The script auto-detects CUDA, MPS, or CPU. On a single GPU it runs without modification. Periodic validation loss, HellaSwag accuracy, and generation samples are logged to `log/log.txt`, with checkpoints saved every 5,000 steps.

## Hardware

Development is done on a single RTX 4070 Super (12 GB), which is sufficient for the TurboQuant inference work since quantization is applied at decode time on a frozen model. Full pretraining of GPT-2 124M on 10B tokens is impractical at this scale and is run on a remote 8x A100 (80 GB) node via [Lambda Cloud](https://lambda.ai/). Single-GPU training is supported for short sanity-check runs.

## Roadmap

| Phase | Component | Status |
| --- | --- | --- |
| 0 | GPT-2 124M from-scratch implementation | Complete |
| 0 | FineWeb-Edu training pipeline | Complete |
| 0 | HellaSwag zero-shot evaluation | Complete |
| 1 | KV cache and incremental decoding | In progress |
| 2 | Per-channel uniform INT4/INT8 baseline | Planned |
| 3 | Randomized Hadamard rotation | Planned |
| 4 | Lloyd-Max codebook quantization | Planned |
| 5 | QJL 1-bit KV cache | Planned |
| 6 | End-to-end ablation report | Planned |

The reported metrics will be perplexity on a held-out FineWeb-Edu shard, HellaSwag accuracy, and cache memory footprint as a function of bit-width, all relative to a BF16 baseline.

## References

**Foundations**

- Vaswani et al., *Attention Is All You Need*. arXiv:1706.03762. https://arxiv.org/pdf/1706.03762
- Rush et al., *The Annotated Transformer*. Harvard NLP. https://nlp.seas.harvard.edu/annotated-transformer/
- Hugging Face, *LLM Course, Chapter 1.4: How do Transformers work?* https://huggingface.co/learn/llm-course/chapter1/4
- Karpathy, *Let's reproduce GPT-2 (124M)*. YouTube. https://www.youtube.com/watch?v=l8pRSuU81PU&t=13272s

**TurboQuant and KV cache quantization lineage**

- Google Research, *TurboQuant: Redefining AI Efficiency with Extreme Compression*. https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- Zandieh et al., *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. arXiv:2504.19874. https://arxiv.org/abs/2504.19874
- Han et al., *PolarQuant: Quantizing KV Caches with Polar Transformation*. arXiv:2502.02617. https://arxiv.org/abs/2502.02617
- Zandieh, Daliri, Han, *QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead*. arXiv:2406.03482. https://arxiv.org/abs/2406.03482

**Dataset and infrastructure**

- *FineWeb-Edu*. Hugging Face Datasets. https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- *Lambda Cloud GPU*. https://lambda.ai/

## License

MIT.
