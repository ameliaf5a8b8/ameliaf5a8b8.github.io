---
title: Unbiased Step Size trick
date: 2026-09-26
lastmod: 2026-09-26
tags:
  - Medium
categories:
  - Machine Learning
  - Reinforcement Learning
  - Math
math: true
draft: false
mathEngine: mathjax
summary: The Unbiased Step Size Trick used to eliminate initial biases
---
$$\text{let }  \alpha_n = \frac{\beta}{\bar{O}_n} \quad \forall n > 0 ,\bar{O}_0 =  0$$
Where $\bar O_n$ follows the recursive relation:
$$ \bar{O}_{n} = \bar{O}_{n-1} + \beta(1 - \bar{O}_{n-1}) , \bar{O}_0 = 1$$
$\bar{R_{t}}$ follows the recursive relation $\bar{R}_{t+1} = \bar{R}_{t} + \delta \alpha_{t}$. Using $R_t - \bar{R}_t$ as the td error:
$$\begin{align}
\bar{R}_{t+1}
&= \bar{R}_t + (R_t - \bar{R}_t)\alpha_t \\
&= \alpha_t R_t + (1 - \alpha_t)\bar{R}_t \\
&= \alpha_t R_t
  + \alpha_{t-1}(1-\alpha_t)R_{t-1}
  + (1-\alpha_t)(1-\alpha_{t-1})\bar{R}_{t-2} \\
&= \sum_{k=1}^{t} \alpha_k R_k
   \prod_{i=k}^{t}(1-\alpha_i)\\
&= \sum_{k=1}^{t} \alpha_k R_k
   \prod_{i=k+1}^{t}(1-\alpha_i)
   + \bar{R}_1\prod_{i=2}^{t}(1-\alpha_i).
\end{align}$$

When $\alpha_0 = 2$, the coefficient of $R_1$ is 0, hence there is no initial bias.

As $n \rightarrow \infty$, $\alpha_n \rightarrow \beta$.
$$\begin{align}
\bar{O}_n
&= \bar{O}_{n-1} + \beta(1 - \bar{O}_{n-1}) \\
&= \beta + (1-\beta)\bar{O}_{n-1} \\
&= \beta + \beta(1-\beta)
   + (1-\beta)^2\bar{O}_{n-2} \\
&= \beta\sum_{i=0}^{n-1}(1-\beta)^i
   + (1-\beta)^n\bar{O}_0 

\end{align}$$$$
\begin{align}
\lim_{ n \to \infty }
   \beta\sum_{i=0}^{\infty}(1-\beta)^i 
&= \frac{\beta}{1-(1-\beta)} \\
&= 1.
\end{align} $$
