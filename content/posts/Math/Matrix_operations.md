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
$$\begin{gather*}(A + B)^T = A^T + B^T \\
(AB)^T = B^TA^T \end{gather*}$$
# Trace
The trace function is linear and has the cyclic property.
$$\begin{gather*}
\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B) \\ 
\text{tr}(AB) = \text{tr}(BA)
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
# Calculus

Differentiation and Integration are linear operators.
$$\begin{gather*} 
\frac{d}{dt}(A + B) = \frac{dA}{dt} + \frac{dB}{dt}\\
\int (A + B) dt= \int A \,dt + \int B \,dt
\end{gather*}$$
## Differentiation

$$\begin{gather*}\frac{d}{dx}(A B) = \frac{dA}{dx} B + A \frac{dB}{dx}\\ 
\begin{aligned}\frac{d}{dx}(ABC) &= \frac{dA}{dx} BC  +A \frac{dBC}{dx}\\
&= \frac{dA}{dx} BC  + A \left(\frac{dB}{dx} C + B \frac{dC}{dx}  \right)\\
&=\frac{dA}{dx}BC + A\frac{dB}{dx}C + AB\frac{dC}{dx}
\end{aligned}
\end{gather*}$$


