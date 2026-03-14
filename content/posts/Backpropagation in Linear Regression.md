---
title: "Backpropagation in Linear Regression"
date: 2026-03-13
lastmod: 2026-03-14
tags: ["Backpropagation", "Linear Algebra", "Jacobian","Matrix Calculus"]
categories: ["Machine Learning",Math]
math: true
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
&= 2(\hat{y}_1 - y_1) \times \begin{bmatrix} \frac{\partial}{\partial w_{11}} & \frac{\partial}{\partial w_{12}} \\ \frac{\partial}{\partial w_{21}} & \frac{\partial}{\partial w_{22}} \end{bmatrix}  (w_{11}x_1 + w_{12}x_2 + b_1- y_1)\\
 &= 2(\hat{y}_1 - y_1) \times \begin{bmatrix} \frac{\partial}{\partial w_{11}}(w_{11}x_1 + w_{12}x_2 + b_1- y_1) & \frac{\partial}{\partial w_{12}}(w_{11}x_1 + w_{12}x_2 + b_1- y_1) \\ \frac{\partial}{\partial w_{21}}(w_{11}x_1 + w_{12}x_2 + b_1- y_1) & \frac{\partial}{\partial w_{22}}(w_{11}x_1 + w_{12}x_2 + b_1- y_1) \end{bmatrix}\\
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
We can also calculate the Jacobian Matrix through matrix calculus operations.
$$\begin{align*}
\hat{\vec y} &= W \vec x + \vec b \\
             &= \begin{bmatrix} 
                w_{11} & w_{12} \\ 
                w_{21} & w_{22} 
                \end{bmatrix}
                \begin{bmatrix} 
                x_1 \\ 
                x_2 
                \end{bmatrix}
                +
                \begin{bmatrix} 
                b_1 \\ 
                b_2 
                \end{bmatrix} \\
L &= \frac{1}{n}\lvert \hat{\vec y} - \vec y \rvert_2
\end{align*}$$
We want to find
$$\frac{\partial L}{\partial W}$$
From the chain rule <span id="eqn:1"></span>$$\text d L = \left( \frac{\partial L}{\partial \vec{y}} \right)^\top \text d\vec{y} \tag 1$$As we are taking the partial derivative w.r.t $W$, we treat $\vec x$ and $\vec b$ as constants.
$$\begin{align*}\text d \vec y &= \text d(W) \vec x
\end{align*}$$
As $dL$ is a scalar, it is equal to its trace. Let's substitute into [(1)](#eqn:1)
$$\begin{align*}
\text d L &= \left( \frac{\partial L}{\partial \vec y} \right)^\top \text d (W) \vec x \\
&= \text{Tr} \left( \left( \frac{\partial L}{\partial \vec y} \right)^\top \text d (W) \vec x \right) \\
&= \text{Tr} \left( \vec x \left( \frac{\partial L}{\partial \vec y} \right)^\top \text d (W) \right) \\
&= \text{Tr} \left( \left[ \frac{\partial L}{\partial \vec y} \vec x^\top \right]^\top \text d (W) \right)
\end{align*}$$

The formal definition of a matrix gradient is based on the [Frobenius Inner Product]({{% relref "Math/Matrix_operations/#trace" %}}):<span id="eqn:2"></span>
$$\begin{align*} \text dL &= \langle \nabla_W L, \text dW \rangle_F \\
&= \text{Tr}\left( (\nabla_W L)^\top \text dW \right)\\
&= \text{Tr}\left( \left( \frac{\partial L}{\partial W} \right)^\top \text dW \right)
\end{align*} \tag 2$$
By comparing our equations
<span id="eqn:3"></span>
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \vec{y}} \vec{x}^\top \tag{3}$$
Focusing on $\frac{\partial L}{\partial \vec y}$ 
$$\begin{align*}
L &= \frac{1}{n}\lvert \hat{\vec y} - \vec y \rvert_2 \\
&= \frac{1}{n}\lvert r \rvert_{2}
\end{align*}$$
where $r = \hat{\vec{y}} - \vec{ y}$ is the *residual*.
$$\begin{align*}
{\partial L} &= \frac{1}{n}\partial(r^\top r)\\
				&= \frac{1}{n}r^\top \partial r + \partial r^\top r \\
				&= \frac{1}{n}r^\top \partial r + \left( r^\top  \partial  r \right)^\top\\
				&=\frac{1}{n} 2r^\top \partial r
						\end{align*}$$
As we want to compute $\frac{\partial L}{\partial \vec y}$, we hold $\hat{\vec{y}}$ constant.
$$\begin{align*}
\partial r &= \partial \hat{\vec{y}} - \partial \vec{y}\\
&=\partial {\vec y}
\end{align*}$$
Substituting
$$\begin{gather*}
\partial L = \frac{1}{n}2r^\top \partial \vec{ y}\\
	\frac{\partial L}{\partial W} =\frac{1}{n} 2r^\top \vec{x}^\top 
\end{gather*}$$
# Derivation of equations [(1)](#eqn:1) and [(2)](#eqn:2)
Equation 1
$$\begin{align*}
\text d L &= \begin{bmatrix} \frac{\partial L}{\partial y_1} & \cdots & \frac{\partial L}{\partial y_n} \end{bmatrix} \begin{bmatrix} \text d y_1 \\ \vdots \\ \text d y_n \end{bmatrix} \\
&= \left( \frac{\partial L}{\partial \vec y} \right)^\top \text d \vec y
\end{align*}$$
Equation 2
Let $A = \frac{\partial L}{\partial W}$ and $B = dW$. For a $2 \times 2$ system, these matrices look like this:

$$A = \begin{bmatrix} \frac{\partial L}{\partial w_{11}} & \frac{\partial L}{\partial w_{12}} \\ \frac{\partial L}{\partial w_{21}} & \frac{\partial L}{\partial w_{22}} \end{bmatrix}, \quad B = \begin{bmatrix} \text d w_{11} & \text d w_{12} \\ \text d w_{21} & \text d w_{22} \end{bmatrix}$$
We want to find $\text d L$, which is the sum of every partial derivative times its specific nudge:
$$dL = \frac{\partial L}{\partial w_{11}}dw_{11} + \frac{\partial L}{\partial w_{12}}dw_{12} + \frac{\partial L}{\partial w_{21}}dw_{21} + \frac{\partial L}{\partial w_{22}}dw_{22}$$
By applying $A^\top B$, we notice
$$
\begin{align*}
A^\top B &= \begin{bmatrix} \frac{\partial L}{\partial w_{11}} & \frac{\partial L}{\partial w_{21}} \\ \frac{\partial L}{\partial w_{12}} & \frac{\partial L}{\partial w_{22}} \end{bmatrix} \begin{bmatrix} dw_{11} & dw_{12} \\ dw_{21} & dw_{22} \end{bmatrix}\\
 &= \begin{bmatrix} \left( \frac{\partial L}{\partial w_{11}}dw_{11} + \frac{\partial L}{\partial w_{21}}dw_{21} \right) & \dots \\ \dots & \left( \frac{\partial L}{\partial w_{12}}dw_{12} + \frac{\partial L}{\partial w_{22}}dw_{22} \right) \end{bmatrix}
\end{align*}
$$
Through the trace, we can extract the sum of the diagonal elements.

$$
\begin{align*}
\text{Tr}(A^\top B) =& \left( \frac{\partial L}{\partial w_{11}}dw_{11} + \frac{\partial L}{\partial w_{21}}dw_{21} \right) + \left( \frac{\partial L}{\partial w_{12}}dw_{12} + \frac{\partial L}{\partial w_{22}}dw_{22} \right)\\
=& \frac{\partial L}{\partial w_{11}}dw_{11} + \frac{\partial L}{\partial w_{12}}dw_{12} + \frac{\partial L}{\partial w_{21}}dw_{21} + \frac{\partial L}{\partial w_{22}}dw_{22}\\
&= \text{d}L
\end{align*}
$$