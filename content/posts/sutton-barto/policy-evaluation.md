---
title: "MD"
date: 2026-01-15
lastmod: 2026-02-05
tags: []
categories: ["reinforcement-learning", "sutton-barto"]
math: true
---
# Introduction

#### Problem.

Let $\lbrace R_n\rbrace_\lbrace n \ge 1 \rbrace$ be a sequence of rewards and let $\alpha \in (0,1)$ be a constant step size. Consider the recursive update, for $n \ge 1$, $$\begin\lbrace equation \rbrace
Q_\lbrace n+1 \rbrace = Q_n + \alpha \bigl(R_n - Q_n\bigr),
\qquad Q_1 = 5, \label\lbrace eq:original \rbrace
\end\lbrace equation \rbrace$$ where $Q_n \in \mathbb\lbrace R \rbrace$ for all $n$.

Define the auxiliary sequence $\\lbrace \bar\lbrace o \rbrace_n\ \rbrace_\lbrace n \ge 0 \rbrace$ by $$\begin\lbrace equation \rbrace
\bar\lbrace o \rbrace_n \doteq \bar\lbrace o \rbrace_\lbrace n-1 \rbrace + \alpha \bigl(1 - \bar\lbrace o \rbrace_\lbrace n-1 \rbrace\bigr),
\quad \text\lbrace for  \rbrace n > 0,
\qquad \bar\lbrace o \rbrace_0 \doteq 0,
\end\lbrace equation \rbrace$$ and let $$\begin\lbrace equation \rbrace
\beta_n \doteq \alpha \bar\lbrace o \rbrace_n.
\end\lbrace equation \rbrace$$

Considering the modified update $$\begin\lbrace equation \rbrace
Q_\lbrace n+1 \rbrace = Q_n + \beta_n (R_n - Q_n).
\end\lbrace equation \rbrace$$ show that the sequence $\\lbrace Q_n\ \rbrace$ can be written as an exponentially recency-weighted average of the rewards $\\lbrace R_k\ \rbrace$. In addition, show that normalizing by $\bar\lbrace o \rbrace_n$ removes the bias introduced by the initialization $Q_1 = 5$.

#### Solution.

Substituting $\alpha$ as $\beta_n$ $$\begin\lbrace align \rbrace
Q_\lbrace n+1 \rbrace &= Q_n + \beta_n \bigl(R_n - Q_n\bigr) \tag\lbrace from \eqref\lbrace eq:original \rbrace \rbrace\\
&=\beta_nR_n + \left[1 - \beta_n \right] Q_n\\
&=\left[ \sum_\lbrace i=1 \rbrace^n\beta_iR_i \prod_\lbrace j=i+1 \rbrace^n 1-\beta_j\right] + Q_1 \prod_\lbrace i=0 \rbrace^n 1- \beta_n
\end\lbrace align \rbrace$$

