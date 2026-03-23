---
title: Matrix operations
date: 2026-03-14
lastmod: 2026-03-14
tags:
  - Easy
categories:
  - Math
math: true
summary:
draft: false
---
# Scalar Multiple
The scalar multiple is a linear function.
$$k(A+B) = kA + kB$$
# Transpose
The transpose function is both linear and has the multiplicative property.
$$\begin{gather*}(A + B)^\top  = A^\top  + B^\top  \\
(AB)^\top  = B^\top A^\top  \end{gather*}$$
# Trace
The trace function is linear and has the cyclic property.
$$\begin{gather*}
\text{Tr}(A + B) = \text{Tr}(A) + \text{Tr}(B) \\ 
\text{Tr}(ABC) = \text{Tr}(CBA)
\end{gather*}$$
# Inverse
The inverse function is multiplicative, but **not** linear. 
$$\begin{gather*}(AB)^{-1} = B^{-1} A^{-1}\\
(A+B)^{-1} \neq A^{-1} + B^{-1}\end{gather*}$$
# Determinant
The determinant is another multiplicative **non-linear** function.
$$\begin{gather*}\det(AB) = \det(A)\det(B)\\
\det(cA) = c^n \det(A)\end{gather*}
$$
# Inner products
Inner products are *sesquilinear* functions which exhibit **conjugate symmetry**.
$$\langle a, b\rangle = \overline{\langle b,a\rangle}$$
where $a$ and $b$ are either both vectors or matrices.

They are also **positive definite**.
$$\begin{gather*}
\langle v, v\rangle \geq 0 \quad \forall v \neq0\\
\langle v, v\rangle = 0 \quad \text{if } v = 0
\end{gather*}$$
where $v$ can either be a vector or matrix.
## Vectors
In the real vector space, inner products are commonly denoted as dot products. 
For the complex space, they are linear in the first argument, and conjugate-linear in the other.
$$\begin{gather*}
\langle a + b, c \rangle = \langle a, c \rangle + \langle b, c \rangle \\
\langle a, b + c \rangle = \langle a, b \rangle + \langle a, c \rangle 
\end{gather*}$$$$\begin{gather*}
\langle ka, c \rangle = k \langle a, c \rangle \\
\langle a, kb \rangle = \bar{k} \langle a, b \rangle
\end{gather*}$$

## Matrices {#sec:frobenius_inner_product}
The **Frobenius inner product** is used for matrices. It is the sum of the element wise product between matrices.
$$ \langle A,B\rangle_F = \sum (A \odot B) = \text{Tr}(A^T B) $$
where $A$, $B \in \mathbb R^{n\times n}$ 

Like vectors, the Forbenius product is linear in the first argument, and conjugate-linear in the other.
$$\begin{gather*}
\langle A + B, C \rangle = \langle A, C \rangle + \langle B, C \rangle \\
\langle A, B + C \rangle = \langle A, B \rangle + \langle A, C \rangle 
\end{gather*}$$$$\begin{gather*}
\langle kA, C \rangle = k \langle A, C \rangle \\
\langle A, kB \rangle = \bar{k} \langle A, B \rangle
\end{gather*}$$
# Norms

# $L_1$  

The $L_1$ norm, also known as the Manhattan distance, measures the magnitude of a vector by summing the absolute values of its components.
$$\|x\|_1 = \sum^{n}_{i=1}|x_i|$$
For matrices, the L1 norm is calculated by
- Taking the L1 norm of each column
- Finding the highest value

## $L_2$
The $L_2$ norm, or Euclidean distance, measures the magnitude of a vector by computing the square root of the sum of squared components.
$$\|x\|_2 = \sqrt{\sum^{n}_{i=1}x_i^2}$$
It is congruent to the square root of the inner product of a vector with itself.  

The Frobenius norm is the matrix equivalent of the Euclidean norm for vectors. It is calculated by flattening the matrix into a vector, and applying the L2 norm above.
 It is also the square root of the [Frobenius inner product](#sec:frobenius_inner_product) of the matrix with itself.

## $L_\infty$

<div class="info-box">
<strong>TODO</strong></div>

# Calculus

Differentiation and Integration are linear operators.
$$\begin{gather*} 
\frac{\text d}{\text dt}(A + B) = \frac{\text dA}{\text dt} + \frac{\text dB}{\text dt}\\
\int (A + B) \text dt= \int A \,\text dt + \int B \,\text dt
\end{gather*}$$
## Differentiation

When differentiating with matrices only, most of the rules from scalar differentiation carry over.
$$\begin{gather*}\text d(A B) = \text dA B + A \text dB\\ 
\begin{aligned}\text d(ABC) &= (\text dA) BC  +A (\text dBC)\\
&=  (\text dA) BC  + A \left[ (\text dB) C + B  (\text dC)  \right]\\
&= (\text dA)BC + A (\text dB)C + AB (\text dC)
\end{aligned}
\end{gather*}$$



