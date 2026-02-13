---
title: "Deriving Backpropagation Gradients"
date: 2026-02-13
lastmod: 2026-02-13
tags: []
categories: []
math: true
summary:
---
# Deriving the gradients for a single value

Let us derive the gradients that happen in backpropogation, one value at a time.

$$\text{let }\theta, W\in \mathbb R^n \text{ and } n \text{ be an arbitrary value}$$

<figure id="fig:1" data-latex-placement="!h">

<figcaption>A typical Neural Net</figcaption>
</figure>

# Adding the loss derivative

Let $\mathcal L$ denote the loss. The update rule uses the gradient of the loss with respect to the variable we are tuning:
$$
\begin{align*}
W_{h_1} 
&\leftarrow W_{h_1} - \alpha \frac{\partial \mathcal L}{\partial W_{h_1}} 
\end{align*}
$$

To find $\frac{\partial \mathcal L}{\partial W_{h_1}}$, we can expand it with the chain rule
$$
\begin{align*}
\frac{\partial \mathcal L}{\partial W_{h_1}} &= \frac{\partial \mathcal L}{\partial o}\frac{\partial o}{\partial W_{h_1}}\\
&=
    \left(
\frac{\partial \mathcal L}{\partial o_1}
\frac{\partial o_1}{\partial W_{h_1}}
+ \dots +
\frac{\partial \mathcal L}{\partial o_n}
\frac{\partial o_n}{\partial W_{h_1}}
\right) 
\end{align*}
$$
Similarly for the bias:
$$
\begin{align*}
b_{h_1}
&\leftarrow b_{h_1}
- \alpha
\left(
\frac{\partial \mathcal L}{\partial o_1}
\frac{\partial o_1}{\partial b_{h_1}}
+ \dots +
\frac{\partial \mathcal L}{\partial o_n}
\frac{\partial o_n}{\partial b_{h_1}}
\right) \\
&= b_{h_1} - \alpha \frac{\partial \mathcal L}{\partial b_{h_1}}
\end{align*}
$$
# Deriving gradients in the vector case

$$\begin{gather*}
\text{let }
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
\end{gather*}$$ 
By the update rule: 
$$
W_h \leftarrow W_{h} - \alpha \frac{\partial \mathcal L}{\partial W_h}
$$


$$
\begin{align*}
\frac{\partial \mathcal L}{\partial W_h} &\doteq \begin{pmatrix}
\frac{\partial \mathcal L}{\partial W_{h_1}} \\
\vdots \\
\frac{\partial \mathcal L}{\partial W_{h_n}}
\end{pmatrix} \\
&= \begin{pmatrix}
\frac{\partial o_1}{\partial W_{h_1}} + \dots + \frac{\partial o_n}{\partial W_{h_1}} \\
\vdots \\
\frac{\partial o_1}{\partial W_{h_n}} + \dots + \frac{\partial o_n}{\partial W_{h_n}}
\end{pmatrix}
\end{align*}
$$
# Deriving gradients in earlier layers

We are trying to find $\frac{\partial \mathcal L}{\partial W_i}$. Through the chain rule, we can expand this into:

$$\frac{\partial \mathcal{L}}{\partial W_i} =  \frac{\partial \mathcal L}{\partial o}\frac{\partial o}{\partial W_h}\frac{\partial W_h}{\partial W_i}$$

