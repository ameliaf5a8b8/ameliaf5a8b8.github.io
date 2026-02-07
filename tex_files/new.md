---
title: "GRUs"
date: 2026-02-08
lastmod: 2026-02-08
tags: []
categories: []
math: true
summary: 
---
# Introduction

<figure id="fig:example" data-latex-placement="h">
<img src="GRU diagram.png" style="width:60.0%" />
<figcaption>GRU block</figcaption>
</figure>

GRU stands for Gated Recurrent Unit, which is a type of recurrent neural network (RNN) that is based on Long Short-Term memory (LSTM). Like LSTM, GRU is designed to model sequential data by allowing information to be selectively remembered or forgotten over time. However, GRU has a simpler architecture than LSTM, with fewer parameters, which makes it easier to train at a cost of accuracy.

The main difference between GRU and LSTM is the way they handle their long term memory through the memory cell state. In LSTM, the memory cell state is maintained separately from the short-term memory, which is represented by the hidden state, and is updated using three gates: the input gate, output gate, and forget gate. In GRU, the short and long term memory are combined to form one hidden state, with the long-term memory is replaced with a "candidate activation vector," which is updated using two gates, the reset gate $r$ and update gate $z$.

The reset gate determines how much of the previous hidden state to forget, while the update gate determines how much of the candidate activation vector to incorporate into the new hidden state.

Overall, GRU is a popular alternative to LSTM for modeling sequential data, due to it's simpler architecture reducing the computational resources required.

# How do GRUs work?

Like other recurrent neural network architectures, GRU processes sequential data one element at a time, updating its hidden state $h\_t \in \mathbb{R}^N$ based on the current input $x\_t \in \mathbb R^N$ and the previous hidden state, where $N$ is the number of features in the input $x\_t$. At each time step, the GRU computes a "candidate activation vector" $z\_t \in \mathbb R^N$ that combines information from the input and the previous hidden state. This candidate vector is then used to update the hidden state for the next time step.

The candidate activation vector is computed using two gates: the reset gate and the update gate. The reset gate determines how much of the previous hidden state to forget, while the update gate determines how much of the candidate activation vector to incorporate into the new hidden state.

**Here's the math behind it:**

The output of the reset and update gate are both computed with the current input $x\_t$ and the previous hidden state $h\_{t-1}$. $$r\_t = \sigma\left(W\_r \odot \left[h\_{t-1},x\_t \right]\right)$$ Where $\sigma$ is the sigmoid function, and $W\_r \in \mathbb R^{2n}$ and $W\_z \in \mathbb R^{2n}$ are weight matrices to be learnt during training.

The candidate activation vector $h^\prime\_t$ is computed using the current input $x\_t$ and a modified version of the previous hidden state that is \"reset\" by the reset gate:

$$h^\prime\_t = \tanh{\left[W\_c \odot \left[r\_t \odot h\_{t-1}, x\_t\right]\right]}$$

The new hidden state $h\_t$ is calculated by adding the candidate action vector $h^\prime\_t$ to the previous hidden state $h\_{t-1}$ , weighted by the update gate.

$$h\_t = (1 - z\_t) \odot h\_{t-1} + z\_t \odot h^\prime\_t$$

# Conclusions

GRU networks are similar to Long Short-Term Memory (LSTM) networks, but with fewer parameters to train, making them a powerful tool for modeling sequential data in cases where computational resources are limited or where a simpler architecture is desired. However, they may not perform as well as LSTMs on tasks that require modeling very long-term dependencies or complex sequential patterns, and are more prone to overfitting than LSTMs, on smaller datasets.