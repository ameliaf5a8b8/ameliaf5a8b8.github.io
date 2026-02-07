---
title: "GRUs"
date: 2026-02-07
lastmod: 2026-02-07
tags: ["Machine Learning", "RNNs"]
categories: []
math: true
summary: GRU stands for Gated Recurrent Unit, which is a type of recurrent neural network (RNN) that is based on Long Short-Term memory (LSTM). Like LSTM, GRU is designed to model sequential data by allowing information to be selectively remembered or forgotten over time. However, GRU has a simpler architecture than LSTM, with fewer parameters, which makes it easier to train at a cost of accuracy
---

# Introduction

<figure id="fig:example" data-latex-placement="h">
  <img src="/images/GRU diagram.png"
       style="width:60%; display:block; margin:auto;" />
  <figcaption style="text-align:center;">GRU block</figcaption>
</figure>

GRU stands for Gated Recurrent Unit, which is a type of recurrent neural network (RNN) that is based on Long Short-Term memory (LSTM). Like LSTM, GRU is designed to model sequential data by allowing information to be selectively remembered or forgotten over time. However, GRU has a simpler architecture than LSTM, with fewer parameters, which makes it easier to train at a cost of accuracy.

The main difference between GRU and LSTM is the way they handle their long term memory through the memory cell state. In LSTM, the memory cell state is maintained separately from the short-term memory, which is represented by the hidden state, and is updated using three gates: the input gate, output gate, and forget gate. In GRU, the short and long term memory are combined to form one hidden state, with the long-term memory is replaced with a "candidate activation vector," which is updated using two gates, the reset gate $r$ and update gate $z$.

The reset gate determines how much of the previous hidden state to forget, while the update gate determines how much of the candidate activation vector to incorporate into the new hidden state.

Overall, GRU is a popular alternative to LSTM for modeling sequential data, due to it's simpler architecture reducing the computational resources required.

# How do GRUs work?

Like other recurrent neural network architectures, GRU processes sequential data one element at a time, updating its hidden state $h_t \in \mathbb{R}^{N_h}$ based on the current input $x\_t \in \mathbb R^{N_x}$ and the previous hidden state, where $N_x$ is the dimensionality of the input and $N_h$ is the dimensionality of the hidden state. At each time step, the GRU computes a “candidate activation vector” $h^\prime_t \in \mathbb R^N$ that combines information from the input and the previous hidden state. This candidate vector is then used to update the hidden state for the next time step.

The candidate activation vector is computed using two gates: the reset gate and the update gate. The reset gate determines how much of the previous hidden state to forget, while the update gate determines how much of the candidate activation vector to incorporate into the new hidden state.

**Here's the math behind it:**

The output of the reset and update gate are both computed with the current input $x\_t$ and the previous hidden state $h\_{t-1}$. $$\begin{align*}
r\_t = \sigma\left(W\_r \odot \left[h\_{t-1},x\_t \right]\right)\\
z\_t = \sigma\left(W\_z \odot \left[h\_{t-1},x\_t \right]\right)
\end{align*}$$ Where $\sigma$ is the sigmoid function, and $W\_r \in \mathbb R^{2n}$ and $W\_z \in \mathbb R^{2n}$ are weight matrices that are learned during training.

The candidate activation vector $h^\prime\_t$ is computed using the current input $x\_t$ and a modified version of the previous hidden state that is \"reset\" by the reset gate:

$$h^\prime\_t = \tanh{\left[W\_c \odot \left[r\_t \odot h\_{t-1}, x\_t\right]\right]}$$

The new hidden state $h\_t$ is calculated by adding the candidate action vector $h^\prime\_t$ to the previous hidden state $h\_{t-1}$ , weighted by the update gate.

$$h\_t = (1 - z\_t) \odot h\_{t-1} + z\_t \odot h^\prime\_t$$

# Conclusions

GRU networks are similar to Long Short-Term Memory (LSTM) networks, but with fewer parameters to train, making them a powerful tool for modeling sequential data in cases where computational resources are limited or where a simpler architecture is desired. However, they may not perform as well as LSTMs on tasks that require modeling very long-term dependencies or complex sequential patterns, and are more prone to overfitting than LSTMs, on smaller datasets.

