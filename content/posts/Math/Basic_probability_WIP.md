---
title: Basic probability
date: 2026-03-12
lastmod: 2026-03-17
tags:
  - Easy
categories:
  - Math
math: true
summary:
---
<!-- <div class="dark warning-box">
  <strong>Whoops:</strong> Dark mode is not supported on this page. Images might not look <i>quite</i> right.
</div> -->

# Probability using Venn diagrams

Venn diagrams provide an intuitive way to view the logical relations between a finite number of sets.
Let $S$ be the sample space of an experiment, where $A$ and $B$ are events in the sample space.
The follow illustrates some simple but important relations.


<div style="display: flex; justify-content: space-between; align-items: flex-start;">
  <div style="width: 55%;">
    $$
    \text{P}(A \cup B) = \text{P}(A) + \text{P}(B) - \text{P}(A \cap B)$$
  </div>

<figure  style="width: 40%; text-align: center; margin:0;">
  <img class="light" src="/images/Basic_probability/light_imgs/II.svg" style="border-radius:0;">
  <img class="dark" src="/images/Basic_probability/dark_imgs/II.svg" style="border-radius:0;">
</figure>
</div>

<div style="display: flex; justify-content: space-between; align-items: flex-start;">
  <div style="width: 55%;">
    $$\text{P}(A \cap B') = \text{P}(A) - \text{P}(A \cap B)$$
  </div>

  <figure  style="width: 40%; text-align: center;">
  <img class="light" src="/images/Basic_probability/light_imgs/III.svg" style="border-radius:0;">
  <img class="dark" src="/images/Basic_probability/dark_imgs/III.svg" style="border-radius:0;">
</figure>
</div>

<div style="display: flex; justify-content: space-between; align-items: flex-start;">
  <div style="width: 55%;">
    $$\text{P}(A \cap B') = \text{P}(A \cup B) - \text{P}(B)$$
  </div>

  <figure  style="width: 40%; text-align: center;">
  <img class="light" src="/images/Basic_probability/light_imgs/IV.svg" style="border-radius:0;">
  <img class="dark" src="/images/Basic_probability/dark_imgs/IV.svg" style="border-radius:0;">
</figure>
</div>

<div style="display: flex; justify-content: space-between; align-items: flex-start;">
  <div style="width: 55%;">
    $$\begin{gather*}\text{P}(A) = \text{P}(A \cap B) + \text{P}(A \cap B')\\
    \text{P}(A' \cap B') = 1 - \text{P}(A \cup B)\end{gather*}$$
  </div>

  <figure  style="width: 40%; text-align: center;">
  <img class="light" src="/images/Basic_probability/light_imgs/V.svg" style="border-radius:0;">
  <img class="dark" src="/images/Basic_probability/dark_imgs/V.svg" style="border-radius:0;">
</figure>
</div>


# Conditional probability
The conditional probability of an event $A$ given that event $B$ has occurred is defined as follows
<span id="eqn:1"></span>
$$
\boxed{P(A\cap B) = P(A \mid B)P(B) \tag 1}
$$
The probability of an event $A$ occurring, given that an event $B$ has occurred, is written
as $P(A \mid B)$. It is fair to assume here that since $B$ has occurred, $P(B) \neq 0$ .
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
  <div style="width: 65%;">
    In the Venn diagram <a href="#fig:1">Figure 1</a>, the sample space is reduced to $B$ only, since $B$ has occurred. 
    That is, $P(A \mid B)$ is the probability of $A$ occurring by considering $B$ as the sample space. 
    $B$ is often called the reduced sample space.
  </div> 

  <figure id="fig:1" style="width: 40%; text-align: center;">
  <img class="light" src="/images/Basic_probability/light_imgs/Conditional_probability.svg" style="border-radius:0;">
  <img class="dark" src="/images/Basic_probability/dark_imgs/Conditional_probability.svg" style="border-radius:0;">
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
P(A\cap B) = P(A \mid B)\,P(B) \tag*{Q.E.D.}
$$


The following identity with 3 variables is a known result from the chain rule of conditional probability:
$$\boxed{P(A\cap B \mid C) \doteq P(A \mid B\cap C​)\, P(B \mid C)}$$

We start by applying the definition from [(1)](#eqn:1) conditioning on $C$
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
&= P(A \mid B\cap C​)\, P(B \mid C) \tag*{Q.E.D.}
\end{align*}
$$
A related identity to ponder over
$$
\begin{gather*}
P(A \mid B \cap C) \doteq \frac{P(A \cap B \mid C)}{P(B \mid C) }  \\
% P(A\cap B \mid C) \doteq P(A\mid B\cap C)\,P(B\mid C) 
\end{gather*}$$