---
title: "main"
date: 2026-02-06
lastmod: 2026-02-06
tags: []
categories: []
math: true
---
# Introduction

#### Problem.

Let $\lbrace R\_n \rbrace\_{n \ge 1}$ be a sequence of rewards and let $\alpha \in (0,1)$ be a constant step size. Consider the recursive update, for $n \ge 1$, $$Q\_{n+1} = Q\_n + \alpha \bigl(R\_n - Q\_n\bigr),
\qquad Q\_1 = 5, $$ where $Q\_n \in \mathbb{R}$ for all $n$.

Define the auxiliary sequence $\lbrace \bar{o}\_n \rbrace\_{n \ge 0}$ by $$\bar{o}\_n \doteq \bar{o}\_{n-1} + \alpha \bigl(1 - \bar{o}\_{n-1}\bigr),
\quad \text{for } n > 0,
\qquad \bar{o}\_0 \doteq 0,$$ and let $$\beta\_n \doteq \alpha \bar{o}\_n.$$

Considering the modified update $$Q\_{n+1} = Q\_n + \beta\_n (R\_n - Q\_n).$$ show that the sequence $\lbrace Q\_n \rbrace$ can be written as an exponentially recency-weighted average of the rewards $\lbrace R\_k \rbrace$. In addition, show that normalizing by $\bar{o}\_n$ removes the bias introduced by the initialization $Q\_1 = 5$.

#### Solution.

Substituting $\alpha$ as $\beta\_n$ $$Q\_{n+1} = Q\_n + \beta\_n \bigl(R\_n - Q\_n\bigr) $$ $$=\beta\_nR\_n + \left[1 - \beta\_n \right] Q\_n$$

