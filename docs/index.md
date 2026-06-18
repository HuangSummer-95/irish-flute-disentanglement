---
layout: default
title: "Irish Flute Style Disentanglement – Math"
---

# 🎶 Mathematical Formulation

## β-VAE Loss Function

$$
\mathcal{L}
= \underbrace{\|x - \hat{x}\|^2}_{\text{Reconstruction}}
+ \beta \cdot
\underbrace{
\left(
-\frac{1}{2} \sum_{j=1}^{d}
\bigl(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \bigr)
\right)
}_{\text{KL Divergence}}
$$

---

## Mutual Information Gap (MIG)

$$
\text{ratio of MIG} = \frac{I(z^{(j^*)}; v) - I(z^{(j^{**})}; v)}{I(z^{(j^*)}; v)}
$$

---

## Disentanglement Score

$$
D = \frac{MI(z_p, y_p) - MI(z_p, y_t)}{MI(z_p, y_p)} + \frac{MI(z_t, y_t) - MI(z_t, y_p)}{MI(z_t, y_t)}
$$

---

> This page is auto-rendered by GitHub Pages with MathJax support.