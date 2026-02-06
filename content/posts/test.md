---
title: "TEST"
date: 2026-01-15
lastmod: 2026-02-05
tags: []
categories: ["reinforcement-learning", "sutton-barto"]
math: true
---
# Introduction

#### Problem.

Let $\lbrace R_n\rbrace_{n \ge 1}$ be a sequence of rewards and let $\alpha \in (0,1)$ be a constant step size. Consider the recursive update, for $n \ge 1$, $$Q_{n+1} = Q_n + \alpha \bigl(R_n - Q_n\bigr),
\qquad Q_1 = 5, $$ where $Q_n \in \mathbb{R}$ for all $n$.

Define the auxiliary sequence $\lbrace \bar{o}_n \rbrace_{n \ge 0}$ by $$\bar{o}_n \doteq \bar{o}_{n-1} + \alpha \bigl(1 - \bar{o}_{n-1}\bigr),
\quad \text{for } n > 0,
\qquad \bar{o}_0 \doteq 0,$$ and let $$\beta_n \doteq \alpha \bar{o}_n.$$

Considering the modified update $$Q_{n+1} = Q_n + \beta_n (R_n - Q_n).$$ show that the sequence $\lbrace Q_n \rbrace$ can be written as an exponentially recency-weighted average of the rewards $\lbrace R_k \rbrace$. In addition, show that normalizing by $\bar{o}_n$ removes the bias introduced by the initialization $Q_1 = 5$.

#### Solution.

Substituting $\alpha$ as $\beta_n$ $$Q_{n+1} = Q_n + \beta_n \bigl(R_n - Q_n\bigr) $$ $$=\beta_nR_n + \left[1 - \beta_n \right] Q_n$$

