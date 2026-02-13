---
title: "Deriving Backpropagation Gradients"
date: 2026-02-09
lastmod: 2026-02-13
tags: []
categories: []
math: true
summary: A simple guide walking through the basic algebra involved in Backpropogation
---



 # Deriving the gradients for a single value

Let us derive the gradients that happen in backpropogation, one value at a time.

$$\text{let }\theta, W\in \mathbb R^n \text{ and } n \text{ be an arbitrary value}$$

<figure>
  <picture>
    <source srcset="/images/ANN_with_labels_dark_mode.png"
            media="(prefers-color-scheme: dark)">
    <img src="/images/ANN_with_labels_light_mode.png"
         style="width:60%; display:block; margin:auto;"
         alt="A typical Neural Net">
  </picture>
  <figcaption style="text-align:center;">A typical Neural Net</figcaption>
</figure>



By the update rule:
$$
W_{h_1} \leftarrow W_{h_1} - \alpha \left( \frac{\partial o_1}{\partial W_{h_1}} + \dots + \frac{\partial o_n}{\partial W_{h_1}} \right) 
$$
$$
= W_{h_1} - \alpha\frac{\partial o}{\partial W_{h_1}}
$$
$$
b_{h_1} \leftarrow b_{h_1} - \alpha \left( \frac{\partial o_1}{\partial b_{h_1}} + \dots + \frac{\partial o_n}{\partial b_{h_1}} \right)
$$
$$
= b_{h_1} - \alpha\frac{\partial o}{\partial b_{h_1}}
$$


# Deriving gradients in the vector case
$$
\begin{gather*}
    W_h \doteq 
    \begin{pmatrix}
        W_{h_1} \\
        \vdots\\
        W_{h_n}
    \end{pmatrix}
    \quad
    b_h \doteq 
    \begin{pmatrix}
        b_{h_1} \\
        \vdots\\
        b_{h_n}
    \end{pmatrix}
    \quad
    O_h \doteq 
    \begin{pmatrix}
        o_{h_1} \\
        \vdots\\
        o_{h_n}
    \end{pmatrix}
\end{gather*}
$$
By the update rule:
$$W_h \leftarrow W_{h} - \alpha \frac{\partial o}{\partial W_h} $$

$$
\begin{align*}
\frac{\partial o}{\partial W_h} &\doteq \begin{pmatrix}
\frac{\partial o}{\partial W_{h_1}} \\
\vdots \\
\frac{\partial o}{\partial W_{h_n}}
\end{pmatrix} \\
&= \begin{pmatrix}
\frac{\partial o_1}{\partial W_{h_1}} + \dots + \frac{\partial o_n}{\partial W_{h_1}} \\
\vdots \\
\frac{\partial o_1}{\partial W_{h_n}} + \dots + \frac{\partial o_n}{\partial W_{h_n}}
\end{pmatrix}
\end{align*}
$$


# Deriving gradients in earlier layers

We are trying to find $\frac{\partial o}{\partial W_i}$. Through the chain rule, we can expand this into:

$$\frac{\partial o}{\partial W_i} = \frac{\partial o}{\partial W_h}\frac{\partial W_h}{\partial W_i}$$