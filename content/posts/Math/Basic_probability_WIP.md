---
title: Basic probability
date: 2026-03-12
lastmod: 2026-03-12
tags:
  - Easy
categories:
  - Math
math: true
summary:
---
# Conditional probability
The conditional probability of an event $A$ given that event $B$ has occurred is defined as follows
<span id="eqn:1"></span>
$$
\boxed{P(A\cap B) = P(A \mid B)P(B) \tag 1}
$$
The probability of an event $A$ occurring, given that an event $B$ has occurred, is written
as $P(A \mid B)$. It is fair to assume here that since $B$ has occurred, $P(B) \neq 0$ .
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
  <div style="width: 70%;">
    In the Venn diagram <a href="#fig:1">Figure 1</a>, the sample space is reduced to $B$ only, since $B$ has occurred. 
    That is, $P(A \mid B)$ is the probability of $A$ occurring by considering $B$ as the sample space. 
    $B$ is often called the reduced sample space.
  </div>

  <figure id="fig:1" style="width: 25%; text-align: center;">
  <img class="light" src="/images/Basic_probability/Conditional_probability_light.svg" style="border-radius:0;">
  <img class="dark" src="/images/Basic_probability/Conditional_probability_dark.svg" style="border-radius:0;">
  <figcaption style="text-align:center;">Figure 1</figcaption>
</figure>
</div>



Thus, $P(A \mid B)$ is the probability of the event $A$ occurring within the sample space $B$, or the ratio of $P(A, B)$ and $P(B)$.  
<span id="eqn:2"></span>
$$
\boxed{P(A \mid B) \doteq \frac{P(A \cap B)}{P(B)} \quad P(B) \neq 0 \tag{2}}
$$
This implies
$$
P(A\cap B) = P(A \mid B)\,P(B)
$$


The following identity is known result from the chain rule of conditional probability:
$$\boxed{P(A\cap B \mid C) \doteq P(A \mid B\cap C​)\, P(B \mid C)}$$

To derive it, we start by applying the definition from [(1)](#eqn:1) conditioning on $C$
$$\begin{align*}
P(A\cap B\mid C)&=\frac{P(A\cap B\cap C)​}{P(C)}\\
&= \frac{P(A\cap (B\cap C))​}{P(C)}
\end{align*}
$$
Using the identity from <a href="#eqn:2">(2)</a> 
$$
P(A\cap B\mid C)=\frac{P(A \cap (B\cap C)​)\,P(B \cap C)}{P(C)}
$$
Using the identity from <a href="#eqn:1">(1)</a>
$$
\begin{align*}P(A\cap B \mid C)&=\frac{P(A \mid (B \cap C)​)\,P(B\cap C)}{P(C)}\\
&= \frac{P(A \mid B\cap C​)\,P(B\cap C)}{P(C)}\\
&= P(A \mid B\cap C​)\, P(B \mid C)
\end{align*}
$$
A related identity to ponder over
$$
\begin{gather*}
P(A \mid B \cap C) \doteq \frac{P(A \cap B \mid C)}{P(B \mid C) }  \\
% P(A\cap B \mid C) \doteq P(A\mid B\cap C)\,P(B\mid C) 
\end{gather*}$$