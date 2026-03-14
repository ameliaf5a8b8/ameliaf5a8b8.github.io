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
## Additivity 
A function is additive if the result of the sum of two inputs is equal to the sum of the results of each input individually. The logarithm is an example of this.
<span id="eqn:1"></span>
$$\log(x_1 + x_2) = \log(x_1) + \log(x_2) \tag 1$$

## 1-Homogeneity
This property states that if you scale the input by a certain factor, the output is scaled by that same factor.

$$f(ax) = a f(x)$$
# Associativity and Distributivity
Due to the rules of [Additivity](##Additivity) and [1-Homogeneity](##1-Homogeneity), linear functions are associative and distributive.
$$\begin{gather*}(A \circ B) \circ C = A \circ (B \circ C)\\
A \circ (B + C) = A \circ B + A \circ C\end{gather*}$$
where $A$, $B$, and $C$ are functions, and $\circ$ is an operator that denotes function composition.

# Appendix
**Exercise 1.** $\;$The logarithm exhibits the additive quality as shown in [equation 1](#eqn:1). Explain why it is not a linear function.  
**Exercise 2.** $\;$Both the derivative and integral operator are linear. Show that this is true.
