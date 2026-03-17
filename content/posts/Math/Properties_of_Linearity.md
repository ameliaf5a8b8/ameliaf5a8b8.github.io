---
title: Properties of Linearity
date: 2026-03-14
lastmod: 2026-03-14
tags:
  - Easy
categories:
  - Math
math: true
summary:
draft: false
---
# Conditions of Linearity
## Additivity with respect to addition
A function is additive if it preserves addition, i.e. the result of applying the function to a sum equals the sum of applying the function to each input individually:
$$f(x_1 + x_2) = f(x_1) + f(x_2) $$

## 1-Homogeneity
This property states that if you scale the input by a certain factor, the output is scaled by that same factor.

$$f(ax) = a f(x)$$
# Associativity and Distributivity
As function composition is associative
$$ (A \circ B) \circ C = A \circ (B \circ C) $$
where $A$, $B$, and $C$ are functions, and $\circ$ is an operator that denotes function composition.

Due to the rules of [Additivity](##Additivity) and [1-Homogeneity](##1-Homogeneity), linear functions are distributive.
$$\begin{gather*}(A \circ B) \circ C = A \circ (B \circ C)\end{gather*}$$


# Appendix
**Exercise 1.** $\;$Both the derivative and integral operators are linear. Show that this is true.
