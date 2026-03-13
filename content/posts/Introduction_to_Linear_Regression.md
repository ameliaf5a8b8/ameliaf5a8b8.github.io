---
title: "Introduction to Linear Regression"
date: 2026-03-13
lastmod: 2026-03-13
tags: ["Linear Algebra", "Jacobian","Matrix Calculus"]
categories: ["Machine Learning",Math]
math: true
draft: true
summary: 
---
Consider a case where we want to predict a student's scores, $y$, based on their intelligence, $x$.  Assuming that a student's score is proportional to the their intelligence, one way we can achieve this is by assigning a weight $w$ to $x$. 
$$\boxed{y= w x}$$
However, this assumes that if both $x_1$ and $x_2$​ are zero, the predicted score is also zero. In practice, this assumption may not hold. There may be a baseline score that does not depend on either of these features. To account for this, we introduce a bias term $b$.
$$\boxed{y= w x + b}$$

We now have two parameters, $w$ and $b$, to tweak in order to refine our estimate. One way to do so would be to recursively reduce each value by its gradient with respect to a loss function, like MSE. [[Loss_functions]]
$$\boxed{\begin{gather*}w \leftarrow w - \frac{\partial L}{\partial w} \\
b \leftarrow b - \frac{\partial L}{\partial b}\end{gather*}} $$
To understand why we subtract the gradient, consider a graph of the loss function as a function of $w$.

<div style="display: flex; justify-content: space-between; align-items: flex-start;">
  <div style="width: 35%;">  
    Our goal is to adjust the value of $w$ such that the loss reaches its minimum. Notice that  the value of $w$ should be reduced when the gradient is positive, and increased $w$ when the gradient is negative, to move in the direction that reduces the loss.
  </div>
  <figure id="fig:1" style="width: 50%; text-align: center;">
    <img class="light figure-img"
        src="\images\Introduction_to_Linear_Regression\tangent_light.svg"
        alt="Graph of loss function with a tangent line">
    <img class="dark figure-img"
        src="\images\Introduction_to_Linear_Regression\tangent_dark.svg"
        alt="Graph of loss function with a tangent line">
    <figcaption style="text-align:center;">
        Graph of loss function with a tangent line  
    </figcaption>
    </figure>
</div>


We chose $w$ arbitrarily here, so the same reasoning applies to the bias term $b$ as well.