# 🎶 Irish Traditional Flute Style Disentanglement

> **A prototype study on separating *who plays* (style) from *what is played* (content) in Irish traditional flute recordings using representation learning and disentangled VAEs.**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-CLAP%20%7C%20Wav2Vec2-yellow)](https://huggingface.co/)

---

## 📌 Project Summary

This project explores **audio representation disentanglement** on a small, real-world dataset of Irish traditional flute performances.

We aim to answer one core question:

> **Can we automatically separate “performance style embeddings” from “musical content embeddings” in monophonic flute recordings?**

Two strategies are compared:

| Approach | Description |
|----------|-------------|
| **Implicit Disentanglement** | β-VAE + KL bottleneck |
| **Explicit Disentanglement** | Dual latent subspaces + adversarial training |

---

## 📌 Project Structure

├── audio/
├── data/
│ └── ITM-Flute-Style6/
├── music_style_disentanglement.ipynb
├── clap_offline/
├── wav2vec2-base-960h_offline/
├── environment.yml
├── requirements.txt
└── README.md

⚠️ **Note**: Full dataset applied in this project is ITM-Flute-Style6, but 10 samples can be used in preliminary testing are stored in /audio

## 🗂 Dataset

- **Dataset**: ITM-Flute-Style6  
- **Samples**: 168 audio recordings  
- **Format**: `.wav`, mono  
- **Factors**:
  - **Player (Style)**: 6 performers  
  - **Tune (Content)**: more than 18 traditional tunes  

⚠️ **Note**: Some tunes are only performed by specific players, creating natural factor coupling.

---

## 🧠 Methodology

### 1️⃣ Feature Extraction

Pretrained audio foundation models are used to extract high-level embeddings:

| Model | Sampling Rate | Embedding Dim |
|----|----|----|
| **CLAP (LAION)** | 48 kHz | 512 |
| **Wav2Vec2 (Facebook)** | 16 kHz | 768 |

Models are downloaded **offline** to ensure reproducibility.

---

### 2️⃣ Implicit Disentanglement (β-VAE)

β-VAE encourages disentanglement by limiting latent capacity

- No supervision on factor semantics
- Different factors naturally occupy different latent dimensions

#### β-VAE Loss Function

$$
\mathcal{L}
= \underbrace{\|x - \hat{x}\|^2}_{\text{Reconstruction}} + \beta \cdot \underbrace{ \left( -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right) \right) }_{\text{KL Divergence}}
$$

#### MIG Evaluation Function

Disentanglement is evaluated using the preserved ratio of **Mutual Information Gap (MIG)**.

$$
\text{ratio of MIG} =
\frac{I(z^{(j^{*})}; v) - I(z^{(j^{**})}; v)}{I(z^{(j^{*})}; v)}
$$

---

### 3️⃣ Explicit Disentanglement (Proposed)

We propose a **dual-branch disentangled β-VAE**:
Audio → Shared Encoder →
                        ├─ z_player (style)
                        └─ z_tune (content)

#### Loss Components
- ✅ Classification loss (Player / Tune)
- ✅ KL divergence
- ✅ **Adversarial disentanglement loss**

Adversarial heads penalize:
- Player latent containing Tune information
- Tune latent containing Player information

---

## 🏋️ Training Strategy

A **two-stage training schedule** is adopted:

| Stage | Objective |
|----|----|
| **Stage 1** | Semantic learning (classification only) |
| **Stage 2** | Add adversarial disentanglement |

> ⚠️ Adding adversarial training too early harms representation learning.

---

## 📊 Results

#### Disentanglement Score

$$
D = \frac{MI(z_p, y_p) - MI(z_p, y_t)}{MI(z_p, y_p)} + \frac{MI(z_t, y_t) - MI(z_t, y_p)}{MI(z_t, y_t)}
$$


### Disentanglement Performance

| Metric | Before Adversarial | After Adversarial |
|----|----|----|
| Player MIG | ~0.29 | — |
| Tune MIG | ~0.11 | — |
| Player Leakage | — | ~0.038 |
| Tune Leakage | — | ~0.012 |
| Disentanglement Score | - | ~**1.94** |
| Classification Loss | ~0.09 | ~0.20 |

✅ Leakage is suppressed **below 0.04**  
✅ Latent spaces remain highly discriminative  

---

### Visualization Evidence

- 🎻 **Same Tune, Different Players** → `z_player` forms clear clusters  
- 🎼 **Same Player, Different Tunes** → `z_tune` forms clear clusters  

This confirms **practical disentanglement**, even under factor coupling.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your_username/irish-flute-disentanglement.git
cd irish-flute-disentanglement
```

### 2️⃣ Create Environment (Recommended)

```bash
conda env create -f environment.yml
conda activate flute-disentangle
```

### Or using pip

```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare Models (Offline)

Pretrained models will be downloaded and stored in /wav2vec2-base-960h_offline and /clap_offline locally to avoid network issues.

### 4️⃣ Run the Notebook

#### ✅ Option 1: Visual Studio Code（Recommended）

1. Open the project folder in VS Code:

```bash
code .
```
2. Install the **Jupyter extension** (if not already installed):
- Extensions → search `Jupyter` → install

3. Select the correct Python interpreter:
- `Ctrl + Shift + P`
- Select **Python: Select Interpreter**
- Choose the one from `flute-disentangle`

4. Open the notebook: music_style_disentanglement.ipynb

5. Run cells sequentially.

✅ **Advantages**:
- Stable kernel control
- Easy dependency debugging
- Excellent for large models (CLAP / Wav2Vec2)

---

#### ⚠️ Option 2: Jupyter Notebook（Alternative）

If you prefer the classic interface:

```bash
jupyter notebook
```

Then open: music_style_disentanglement.ipynb

⚠️ On Windows, kernel switching may be less reliable than VS Code.

---

## 🧩 Key Takeaways

- ✅ CLAP & Wav2Vec2 embeddings already contain separable style/content information
- ✅ β-VAE achieves implicit disentanglement without supervision
- ✅ Explicit adversarial training further suppresses leakage
- ✅ **Perfect disentanglement is unnecessary** for downstream tasks
- ✅ Practical disentanglement is achievable even on small datasets

---

## 📜 License

MIT License © 2026 Yuying_Huang

---

## ✉️ Contact

**Yuying Huang**  
📧 yuying.huang.2019@qq.com  
🔗 [GitHub](https://github.com/HuangSummer-95)