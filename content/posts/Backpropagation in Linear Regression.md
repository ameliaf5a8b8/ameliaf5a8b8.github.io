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
\end{align*}
}$$
However, this assumes that if both $x_1$ and $x_2$​ are zero, the predicted score is also zero. In practice, this assumption may not hold. There may be a baseline score that does not depend on either of these features. To account for this, we introduce a bias term $b$.
$$\boxed{\begin{align*}
y_1 &= w_{11}x_1 + w_{12}x_2  + b_1\\
y_2 &= w_{21}x_1 + w_{22}x_2 + b_2
\end{align*}}$$

Notice how this can be written as a linear transformation.
$$
\mathbf{y} = W \mathbf{x} + \mathbf b \implies 
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
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \begin{bmatrix} \hat{y}_1 - y_1 \\ \hat{y}_2 - y_2 \end{bmatrix}$$

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
\mathbf b_{new} &\doteq \mathbf b_{old} - \alpha \frac{\partial L}{\partial \mathbf  b}\\
&= \mathbf b_{old} - \alpha\begin{bmatrix} \hat{y}_1 - y_1 \\ \hat{y}_2 - y_2 \end{bmatrix}
\end{align*}$$
# Appendix
## Derivation of the Weights gradient
We can also calculate the Jacobian Matrix through matrix calculus operations.
$$\begin{align*}
\hat{\mathbf y} &= W \mathbf x + \mathbf b \\
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
L &= \frac{1}{n}\| \hat{\mathbf y} - \mathbf y \|_{2}^2
\end{align*}$$

We want to find
$$\frac{\partial L}{\partial W}$$
From the chain rule <span id="eqn:1.1"></span>$$\text d L = \left( \frac{\partial L}{\partial \mathbf{y}} \right)^\top \text d\mathbf{y} \tag {1.1}$$As we are taking the partial derivative w.r.t $W$, we treat $\mathbf x$ and $\mathbf b$ as constants.
$$\begin{align*}\text d \mathbf y &= \text d(W) \mathbf x
\end{align*}$$
As $\text d L$ is a scalar, it is equal to its trace.
$$\begin{align*}
\text d L &= \left( \frac{\partial L}{\partial \mathbf y} \right)^\top \text d (W) \mathbf x \\
&= \text{Tr} \left( \left( \frac{\partial L}{\partial \mathbf y} \right)^\top \text d (W) \mathbf x \right) \\
&= \text{Tr}   \left(\mathbf x \left( \frac{\partial L}{\partial \mathbf y} \right)^\top \text d W \right) \\
&=\text{Tr}\left( \left[ \frac{\partial L}{\partial \mathbf{y}} \mathbf{x}^\top \right]^\top \text{d}W \right) 
\end{align*}$$

The formal definition of a matrix gradient is based on the [Frobenius Inner Product]({{% relref "Math/Matrix_operations/#trace" %}}):<span id="eqn:1.2"></span>
$$\begin{align*} \text dL &= \langle \nabla_W L, \text dW \rangle_F \\
&= \text{Tr}\left( (\nabla_W L)^\top \text dW \right)\\
&= \text{Tr}\left( \left( \frac{\partial L}{\partial W} \right)^\top \text dW \right)
\end{align*} \tag {1.2}$$
By comparing our equations
<span id="eqn:1.3"></span>
$$\frac{\partial L}{\partial W} =\frac{\partial L}{\partial \mathbf{y}}  \mathbf{x}^\top \tag{1.3}$$
Focusing on the differential of MSE, $\frac{\partial L}{\partial \mathbf y}$ 
$$\begin{align*}
L &= \frac{1}{n}\| \hat{\mathbf y} - \mathbf y \|_{2}^2\\
&= \frac{1}{n}\| \mathbf{r} |_{2}^2
\end{align*}$$
where $\mathbf r = \hat{\mathbf{y}} - \mathbf{ y}$ is the *residual*. 
As we are differentiating with respect to $\mathbf{y}$, we treat $\hat{\mathbf{y}}$ as a constant.
$$\text{d} \mathbf{y} = \text{d} \mathbf{r}$$
Computing the differential
$$\begin{align*} \text{d} L &= \frac{1}{n} \text{d} (\mathbf{r}^\top \mathbf{r}) \\ &= \frac{1}{n} (\text{d} \mathbf{r}^\top \mathbf{r} + \mathbf{r}^\top \text{d} \mathbf{r}) \\ &= \frac{1}{n} (\mathbf{r}^\top \text{d} \mathbf{r} + \mathbf{r}^\top \text{d} \mathbf{r}) \quad \text{since } \mathbf{a}^\top \mathbf{b} = (\mathbf{b}^\top \mathbf{a})^\top \\ &= \frac{2}{n} \mathbf{r}^\top \text{d} \mathbf{r} \\
&= \frac{2}{n}\mathbf{r}^\top \text{d} \mathbf{ y} 
\end{align*}$$
<span id="eqn:1.4"></span>
$$
\frac{\partial L}{\partial \mathbf{y}} = \frac{2}{n}\mathbf{r^\top} \tag {1.4}
$$
Substituting into [equation 1.3](#eqn:1.3)

$$\frac{\partial L}{\partial W} =\frac{2}{n}    \mathbf{r} \mathbf{x}^\top
$$
## Derivation of the bias gradient
To find the gradient of the loss $L$ with respect to the bias $\vec{b}$, we use the chain rule in its differential form. Given $\hat{\vec{y}} = W\vec{x} + \vec{b}$, we treat $W$ and $\vec{x}$ as constants.
$$\begin{equation*}
\text{d}\vec{y} = \text{d}\vec{b}
\end{equation*}$$

Starting from the total differential of the loss
<span id="eqn:2.1"></span>
$$
\begin{align*}
\text{d} L &= \left( \frac{\partial L}{\partial \vec{y}} \right)^\top \text{d}\vec{y} \\
&= \left( \frac{\partial L}{\partial \vec{y}} \right)^\top \text{d}\vec{b} \quad \text{(Substituting } \text{d}\vec{y} = \text{d}\vec{b}\text{)}\tag {2.1}
\end{align*}
$$

From our earlier derivation of the MSE differential in [equation 1.4](#eqn:1.4), we know:

$$\begin{equation*}
\text{d} L = \frac{2}{n} \vec{r}^\top \text{d}\vec{y}
\end{equation*}$$

By comparing this to the general form $\text{d}L = \left( \frac{\partial L}{\partial \vec{y}} \right)^\top \text{d}\vec{y}$, we identify the Jacobian.

$$\begin{equation*}
\left( \frac{\partial L}{\partial \vec{y}} \right)^\top = \frac{2}{n} \vec{r}^\top
\end{equation*}$$

Substituting this back into <a href="#eqn:2.1">equation 2.1</a>

$$\begin{equation*}
\text{d} L = \left( \frac{2}{n} \vec{r}^\top \right) \text{d}\vec{b}
\end{equation*}$$

Finally, we identify the gradient $\frac{\partial L}{\partial \vec{b}}$ by substituting into [equation 2.1](#eqn:2.1)

$$\begin{equation*}
\frac{\partial L}{\partial \vec{b}} = \frac{2}{n} \vec{r}
\end{equation*}$$

# Derivation of equations [(1.1)](#eqn:1.1) and [(1.2)](#eqn:1.2)
Equation 1
$$\begin{align*}
\text d L &= \begin{bmatrix} \frac{\partial L}{\partial y_1} & \cdots & \frac{\partial L}{\partial y_n} \end{bmatrix} \begin{bmatrix} \text d y_1 \\ \vdots \\ \text d y_n \end{bmatrix} \\
&= \left( \frac{\partial L}{\partial \mathbf y} \right)^\top \text d \mathbf y
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
