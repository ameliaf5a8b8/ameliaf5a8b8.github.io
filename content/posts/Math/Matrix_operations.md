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
\text{Tr}(AB) = \text{Tr}(BA)
\end{gather*}$$
It can also be used to calculate the **Frobenius inner product**, which is the sum of the element wise product between matrices.
$$ \langle A,B\rangle_F = \sum (A \odot B) = \text{Tr}(A^T B) $$
where $A$, $B \in \mathbb R^{n\times n}$ 
# Inverse
The inverse function is multiplicative, but **not** linear. 
$$\begin{gather*}(AB)^{-1} = B^{-1} A^{-1}\\
(A+B)^{-1} \neq A^{-1} + B^{-1}\end{gather*}$$
# Determinant
The determinant is another multiplicative **non-linear** function.
$$\begin{gather*}\det(AB) = \det(A)\det(B)\\
\det(cA) = c^n \det(A)\end{gather*}
$$
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



