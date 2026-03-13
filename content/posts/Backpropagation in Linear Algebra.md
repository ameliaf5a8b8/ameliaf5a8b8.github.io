### Linear Transformation and Loss Gradient
Consider a case where we want to predict a student's scores, $y$ through two *features*: their intelligence $x_1$ and their relationship with their parents, $x_2$. One way we can achieve this is by taking a *weighted* sum of $x_1$ and $x_2$.
$$\boxed{y \doteq w_1 x_1 + w_2 x_2}$$
However, this assumes that if both $x_1$ and $x_2$​ are zero, the predicted score is also zero. In practice, this assumption may not hold. There may be a baseline score that does not depend on either of these features. To account for this, we introduce a bias term $b$:
In the scalar form
$$\begin{align*}y_1&= w_{11} x_{11}+ w_{12}x_{12}+ b_1\\
y_2&= w_{21}x_{22} + w_{2}x_{22} + b_2\end{align*}$$
The transformation is defined by:
$$
\vec{y} = W \vec{x} \implies 
\begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = 
\begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix} 
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
$$

The Mean Squared Error (MSE) loss $\mathcal L$ and its partial derivatives are calculated as follows:

$$\begin{align*}
\mathcal L &= \frac{1}{2} \sum_{i=1}^{2} (\hat{y}_i - y_i)^2 \\
\hat{y}_1 &= w_{11}x_1 + w_{12}x_2 \\
\frac{\partial \mathcal L}{\partial W} &= \frac{1}{2} \frac{\partial}{\partial W} \sum_{i=1}^{2} (\hat{y}_i - y_i)^2 \\
&= \frac{1}{2} \left[ \frac{\partial}{\partial W}(\hat{y}_1 - y_1)^2 + \frac{\partial}{\partial W}(\hat{y}_2 - y_2)^2 \right] \\
\end{align*}$$Focusing on $\hat{y}_1$:
$$\begin{align*}
\frac{\partial}{\partial W} (\hat{y}_1 - y_1)^2 &= 2(\hat{y}_1 - y_1) \times \frac{\partial}{\partial W}(w_{11}x_1 + w_{12}x_2 - y_1) \\
&= 2(\hat{y}_1 - y_1)x_1
\end{align*}$$

The resulting Jacobian matrix $J$ (or gradient matrix $\frac{\partial L}{\partial W}$) is:

$$
\frac{\partial \mathcal L}{\partial W} = \begin{bmatrix} 
(\hat{y}_1 - y_1)x_1 & (\hat{y}_1 - y_1)x_2 \\ 
(\hat{y}_2 - y_2)x_1 & (\hat{y}_2 - y_2)x_2 
\end{bmatrix}
$$

---

### Gradient Descent Update Rule

The weight matrix $W$ is updated using the learning rate $\alpha$:

$$
W_{new} = W_{old} - \alpha \frac{\partial L}{\partial W}
$$