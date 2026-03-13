---
title: "Backpropagation in Linear Regression"
date: 2026-03-13
lastmod: 2026-03-13
tags: ["Backpropagation", "Linear Algebra", "Jacobian","Matrix Calculus"]
categories: ["Machine Learning",Math]
math: true
draft: true
summary: A simple guide walking through the basic algebra involved in linear regression
---

# Vector form of a system of equations
Consider a case where we want to predict a student's scores, $y_1$, and their physical fitness, $y_2$,  through two *features*: their intelligence $x_1$, and their relationship with their parents $x_2$. One way we can achieve this is by taking a *weighted* sum of $x_1$ and $x_2$, for each output.
$$\boxed{\begin{align*}
y_1 &= w_{11}x_1 + w_{12}x_2 \\
y_2 &= w_{21}x_1 + w_{22}x_2
\end{align*}}$$
However, this assumes that if both $x_1$ and $x_2$​ are zero, the predicted score is also zero. In practice, this assumption may not hold. There may be a baseline score that does not depend on either of these features. To account for this, we introduce a bias term $b$.
$$\boxed{\begin{align*}
y_1 &= w_{11}x_1 + w_{12}x_2  + b_1\\
y_2 &= w_{21}x_1 + w_{22}x_2 + b_2
\end{align*}}$$

Notice how this can be written as a linear transformation.
$$
\vec{y} = W \vec{x} + \vec b \implies 
\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = 
\begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix} 
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
+
\begin{bmatrix} b_1 \\ b_2 \end{bmatrix}
$$
# First-Principles Derivation of the Jacobian Matrix
The Mean Squared Error (MSE) loss $\mathcal L$ and its partial derivatives are calculated as follows:
$$\begin{align*}
\mathcal L &= \frac{1}{2} \sum_{i=1}^{2} (\hat{y}_i - y_i)^2 \\
\frac{\partial \mathcal L}{\partial W} &= \frac{1}{2} \frac{\partial}{\partial W} \sum_{i=1}^{2} (\hat{y}_i - y_i)^2 \\
&= \frac{1}{2} \left[ \frac{\partial}{\partial W}(\hat{y}_1 - y_1)^2 + \frac{\partial}{\partial W}(\hat{y}_2 - y_2)^2 \right] \\
\end{align*}$$Focusing on $\hat{y}_1$:
$$\begin{align*}\frac{\partial}{\partial W} (\hat{y}_1 - y_1)^2 &= 2(\hat{y}_1 - y_1) \times \frac{\partial}{\partial W} (\hat{y}_1 - y_1)\\
&= 2(\hat{y}_1 - y_1) \times \begin{bmatrix} \frac{\partial}{\partial w_{11}} & \frac{\partial}{\partial w_{12}} \\ \frac{\partial}{\partial w_{21}} & \frac{\partial}{\partial w_{22}} \end{bmatrix} \otimes (w_{11}x_1 + w_{12}x_2 + b_1- y_1)\\
\end{align*}$$
where $\otimes$ is the Kronecker Product
$$\begin{align*}\frac{\partial}{\partial W} (\hat{y}_1 - y_1)^2 &= 2(\hat{y}_1 - y_1) \times \begin{bmatrix} \frac{\partial}{\partial w_{11}}(w_{11}x_1 + w_{12}x_2 + b_1- y_1) & \frac{\partial}{\partial w_{12}}(w_{11}x_1 + w_{12}x_2 + b_1- y_1) \\ \frac{\partial}{\partial w_{21}}(w_{11}x_1 + w_{12}x_2 + b_1- y_1) & \frac{\partial}{\partial w_{22}}(w_{11}x_1 + w_{12}x_2 + b_1- y_1) \end{bmatrix}\\
&= 2(\hat{y}_1 - y_1) \times \begin{bmatrix} x_1 & x_2 \\ 0 & 0 \end{bmatrix} \\
&= \begin{bmatrix}2(\hat{y}_1 - y_1) x_1 & 2(\hat{y}_1 - y_1)x_2 \\ 0 & 0 \end{bmatrix} 
\end{align*} $$

Repeating this for $\hat y_2$
$$\frac{\partial}{\partial W} (\hat{y}_2 - y_2)^2 = \begin{bmatrix} 0 & 0  \\2(\hat{y}_2 - y_2) x_2 & 2(\hat{y}_2 - y_2)x_2 \end{bmatrix} 
 $$


The resulting Jacobian matrix $J$ (or gradient matrix $\frac{\partial L}{\partial W}$) is:

$$
\begin{align*}
\frac{\partial \mathcal L}{\partial W}  &= \frac{1}{2} \left[ \frac{\partial}{\partial W}(\hat{y}_1 - y_1)^2 + \frac{\partial}{\partial W}(\hat{y}_2 - y_2)^2 \right]\\
&= \frac{1}{2} \left(  \begin{bmatrix}2(\hat{y}_1 - y_1) x_1 & 2(\hat{y}_1 - y_1)x_2 \\ 0 & 0 \end{bmatrix} + \begin{bmatrix} 0 & 0  \\2(\hat{y}_2 - y_2) x_2 & 2(\hat{y}_2 - y_2)x_2 \end{bmatrix} \right) \\
&= \frac{1}{2}\begin{bmatrix} 
2(\hat{y}_1 - y_1)x_1 & 2(\hat{y}_1 - y_1)x_2 \\ 
2(\hat{y}_2 - y_2)x_1 & 2(\hat{y}_2 - y_2)x_2 
\end{bmatrix}
\\&=
\begin{bmatrix} 
(\hat{y}_1 - y_1)x_1 & (\hat{y}_1 - y_1)x_2 \\ 
(\hat{y}_2 - y_2)x_1 & (\hat{y}_2 - y_2)x_2 
\end{bmatrix}
\end{align*}
$$

Repeating this for the bias
$$\frac{\partial \mathcal{L}}{\partial \vec{b}} = \begin{bmatrix} \hat{y}_1 - y_1 \\ \hat{y}_2 - y_2 \end{bmatrix}$$

# Gradient Descent Update Rule

The weight matrix $W$ is updated using the learning rate $\alpha \in \mathbb R^+$ 

$$
\begin{align*}
W_{new} &\doteq W_{old} - \alpha \frac{\partial L}{\partial W}\\
&= W_{old} - \alpha \begin{bmatrix} 
(\hat{y}_1 - y_1)x_1 & (\hat{y}_1 - y_1)x_2 \\ 
(\hat{y}_2 - y_2)x_1 & (\hat{y}_2 - y_2)x_2 
\end{bmatrix}
\end{align*}$$

Similarly, for the bias
$$\begin{align*}
\vec b_{new} &\doteq \vec b_{old} - \alpha \frac{\partial L}{\partial \vec  b}\\
&= \vec b_{old} - \alpha\begin{bmatrix} \hat{y}_1 - y_1 \\ \hat{y}_2 - y_2 \end{bmatrix}
\end{align*}$$
# Appendix
this is a todo bye
