[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/attnres-tinyllm/blob/main/attnres_tinyllm.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/arXiv-2603.15031-b31b1b)](https://arxiv.org/abs/2603.15031)
[![SkillWisor](https://img.shields.io/badge/YouTube-SkillWisor-red?logo=youtube)](https://youtube.com/@SkillWisor)

> **From-scratch PyTorch implementation of Attention Residuals (arXiv:2603.15031, Kimi Team 2026)** — replaces fixed residual connections with softmax attention over layer depth. Builds Standard vs AttnRes TinyLLM, trains both, and visualizes hidden norms, depth weights, and loss gaps.

---

## 📄 The Paper

> **Attention Residuals**
> Kimi Team (Moonshot AI) — arXiv:2603.15031 — *Submitted March 2026*
>
> *"Residual connections with PreNorm are standard in modern LLMs, yet they accumulate all layer outputs with fixed unit weights. This uniform aggregation causes uncontrolled hidden-state growth with depth, progressively diluting each layer's contribution. We propose Attention Residuals (AttnRes), which replaces this fixed accumulation with softmax attention over preceding layer outputs, allowing each layer to selectively aggregate earlier representations with learned, input-dependent weights."*
>
> 📎 [Read the paper](https://arxiv.org/abs/2603.15031)

---

## 🧠 The Core Idea in One Picture

```
Standard Residuals (every LLM today):

  h₁ = h₀ + Δ₁          ← weight = 1.0 (fixed)
  h₂ = h₁ + Δ₂          ← weight = 1.0 (fixed)
  h₃ = h₂ + Δ₃          ← weight = 1.0 (fixed)
  ...
  Problem: all layers get equal vote regardless of how useful they are
           hidden state norm grows ∝ √L  (gets worse with depth)


Attention Residuals (this paper):

  hₗ = Σᵢ αᵢ · Vᵢ       ← weights are LEARNED, INPUT-DEPENDENT

  where αᵢ = softmax(proj(RMSNorm(hᵢ)))

  Solution: each layer learns which previous layers to trust
            weights change per token, per context
            hidden state norms stay controlled
```

---

## 📺 Watch the Full Tutorial

> 🎬 **[YouTube — SkillWisor: Implementing a Brand New Research Paper From Scratch](https://youtube.com/@SkillWisor)**
>
> This notebook is the companion code to the video. We go line by line through the paper and implement every equation as runnable code — no skipping the hard parts.

---

## 📋 Table of Contents

| Step | Topic | From Paper |
|------|-------|------------|
| 0 | Setup | — |
| 1 | [The Problem — Hidden State Growth](#step-1) | Section 1 (motivation) |
| 2 | [Data & BPE Tokenizer](#step-2) | — |
| 3 | [Shared Building Blocks](#step-3) | RMSNorm, RoPE, SwiGLU, GQA |
| 4 | [Full AttnRes](#step-4) | Section 3.1 |
| 5 | [Block AttnRes](#step-5) | Section 3.2 |
| 6 | [Standard vs AttnRes Blocks](#step-6) | — |
| 7 | [Full TinyLLM — Both Models](#step-7) | — |
| 8 | [Side-by-Side Training](#step-8) | Section 4 (experiments) |
| 9 | [Analysis — What Changes?](#step-9) | Section 4.3 |
| 10 | [Depth Attention Weights](#step-10) | Figure 3 in paper |
| 11 | [Text Generation](#step-11) | — |
| 12 | [Architecture Diagram](#step-12) | Figure 1 in paper |
| 13 | [Paper Claims Verified](#step-13) | Section 4 |
| 14 | [Save Checkpoint](#step-14) | — |

---

## ⚡ Quickstart

### Option 1 — Google Colab (Recommended)

1. Click **"Open in Colab"** above
2. `Runtime → Run all`
3. Both models train in ~5 minutes on free CPU

### Option 2 — Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/attnres-tinyllm.git
cd attnres-tinyllm

pip install torch numpy matplotlib

jupyter notebook attnres_tinyllm.ipynb
```

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `torch` | Neural network, autograd |
| `numpy` | Numerical operations |
| `matplotlib` | All visualizations |

Zero external ML frameworks. No HuggingFace, no LangChain. Just math and PyTorch.

---

## 🏗️ What We Build

### Two complete TinyLLMs trained side by side

```
TinyLLM-Standard                    TinyLLM-AttnRes
────────────────                    ───────────────
Token Embedding                     Token Embedding
      ↓                                   ↓
StandardBlock × 4                   AttnResBlock × 4
  RMSNorm                             RMSNorm
  GQA + RoPE                          GQA + RoPE
  x = x + attn(x)  ← fixed           h = BlockAttnRes(blocks, partial)
  RMSNorm                             RMSNorm
  x = x + ffn(x)   ← fixed           x = x + ffn(h)  ← learned
      ↓                                   ↓
Final RMSNorm                       Final RMSNorm
LM Head (weight-tied)               LM Head (weight-tied)
```

### Both models use the full modern architecture

| Component | Implementation |
|-----------|---------------|
| Normalization | RMSNorm (not LayerNorm) |
| Positional encoding | RoPE (Rotary Positional Embedding) |
| FFN activation | SwiGLU |
| Attention | Grouped Query Attention (GQA) |
| Optimizer | AdamW with (0.9, 0.95) betas |
| LR schedule | Cosine decay with warmup |

The **only** difference between the two models is the residual connection mechanism.

---

## 🔬 Core Implementation

### Full AttnRes — Paper Section 3.1

```python
def forward(self, layer_outputs: list):
    V      = torch.stack(layer_outputs, dim=0)   # (N, B, T, D)
    K      = self.norm(V)                         # normalize keys
    logits = self.proj(K).squeeze(-1)             # (N, B, T)
    alpha  = F.softmax(logits, dim=0)             # attention over DEPTH
    h      = (alpha.unsqueeze(-1) * V).sum(0)     # weighted sum
    return h, alpha
```

Memory: **O(L × B × T × D)** — stores all L layer outputs.

### Block AttnRes — Paper Section 3.2

```python
def forward(self, completed_blocks: list, partial_block):
    all_reps = completed_blocks + [partial_block]
    V      = torch.stack(all_reps, dim=0)        # (N+1, B, T, D)
    K      = self.norm(V)
    logits = self.proj(K).squeeze(-1)
    alpha  = F.softmax(logits, dim=0)
    h      = (alpha.unsqueeze(-1) * V).sum(0)
    return h, alpha
```

Memory: **O(N × B × T × D)** — N blocks, not L layers. For N=8, L=64: **8x reduction**.

---

## 📊 What the Notebook Measures

| Metric | What it shows |
|--------|---------------|
| Training loss | Standard vs AttnRes learning curve |
| Perplexity | exp(loss) — lower = better |
| Loss gap per epoch | Advantage of AttnRes over time |
| Hidden state norms | Whether norms stay controlled with depth |
| Norm spread (max−min) | Uniformity across layers |
| Depth attention α | Which blocks the model learns to trust |

### Paper claims verified

- ✅ AttnRes achieves lower final loss
- ✅ AttnRes achieves lower perplexity
- ✅ More uniform hidden state norms across depth
- ✅ Non-uniform depth weights (model has learned preferences)
- ✅ Block AttnRes is a drop-in replacement

> **Scale note:** Our model is ~500K params on 100 sentences.
> The paper tests at 48B params on 1.4T tokens.
> Small-scale results may differ in magnitude — the architecture principle is identical.

---

## 🗺️ Paper → Code Mapping

| Paper Section | This Notebook |
|---------------|---------------|
| Section 1 — Motivation | Step 1: Hidden state growth visualization |
| Equation 1 — Standard residual | `StandardBlock.forward()` |
| Equation 2 — Full AttnRes | `FullAttnRes.forward()` — Step 4 |
| Equation 3 — Block AttnRes | `BlockAttnRes.forward()` — Step 5 |
| Figure 1 — Architecture | Step 12: Architecture diagram |
| Figure 2 — Memory comparison | Step 5: Block vs Full memory plot |
| Section 4 — Experiments | Step 8–9: Side-by-side training |
| Figure 3 — Depth weights | Step 10: α visualization |

---

## 🔗 Related Work

| Paper | Connection |
|-------|-----------|
| [Hyper-Connections](https://arxiv.org/abs/2409.19606) | Related approach to dynamic residual weighting |
| [Value Residual Learning](https://arxiv.org/abs/2410.17897) | ACL 2025, similar depth-aggregation motivation |
| [Kimi Linear Architecture](https://arxiv.org/abs/2510.26692) | The 48B model AttnRes was integrated into |
| [RoPE](https://arxiv.org/abs/2104.09864) | Positional encoding used in both models |
| [GQA](https://arxiv.org/abs/2305.13245) | Grouped Query Attention used in both models |

---

## 📁 Repository Structure

```
attnres-tinyllm/
│
├── attnres_tinyllm.ipynb     ← Main notebook (start here)
├── README.md                 ← This file
└── LICENSE                   ← MIT License
```

---

## 🐛 Common Errors & Fixes

**`NameError: name 'x_batch' is not defined`**
Cells ran out of order. Fix: `Runtime → Run all`.

**`NameError: name 'tokenizer' is not defined`**
Same cause — run cells strictly top to bottom.

**`AssertionError: n_heads must be divisible by n_kv_heads`**
Default is N_HEADS=4, N_KV_HEADS=2. If you change these, keep `N_HEADS % N_KV_HEADS == 0`.

---

## 🙋 FAQ

**Q: Is this the official paper code?**
Independent from-scratch implementation based on the paper and official GitHub. The core AttnRes equations match exactly. This is a teaching implementation, not the production Kimi codebase.

**Q: Does AttnRes always beat standard residuals?**
The paper confirms consistent improvement across model sizes in scaling law experiments. At tiny scale gains may be marginal — the bigger the model and dataset, the more pronounced the benefit.

**Q: Can I use this in my own project?**
Yes — MIT licensed. `FullAttnRes` and `BlockAttnRes` are self-contained and drop-in replaceable for standard residuals in any transformer.

**Q: Full AttnRes vs Block AttnRes — which should I use?**
Block AttnRes for anything practical. Full AttnRes is theoretically cleaner but memory cost grows with depth. Block AttnRes with N=8 recovers ~95% of Full AttnRes gains at a fraction of the memory.

---

## 👨‍💻 About

Built by **Naresh Matta** for the **SkillWisor** YouTube channel — implementing cutting-edge AI research papers for Indian builders and working professionals.

- 🎥 YouTube: [@SkillWisor](https://youtube.com/@SkillWisor)
- 💼 LinkedIn: [Naresh Matta](https://linkedin.com/in/nareshmatta)
- 🐙 GitHub: [@inareshmatta](https://github.com/inareshmatta)

---

## 📄 Citation

```bibtex
@article{kimiTeam2026attnres,
  title   = {Attention Residuals},
  author  = {Kimi Team},
  journal = {arXiv preprint arXiv:2603.15031},
  year    = {2026}
}
```

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

<div align="center">

**Implementing papers > reading papers. ⭐ this repo if you agree.**

*Built for SkillWisor — making AI research accessible to Indian builders 🇮🇳*

</div>
