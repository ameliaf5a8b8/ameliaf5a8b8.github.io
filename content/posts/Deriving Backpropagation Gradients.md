---
title: "Deriving Backpropagation Gradients"
date: 2026-02-09
lastmod: 2026-02-09
tags: []
categories: []
math: true
summary: A simple guide walking through the basic algebra involved in Backpropogation
---



 # Deriving the gradients for a single value

Let us derive the gradients that happen in backpropogation, one value at a time.

$$\text{let }\theta, W\in \mathbb R^n \text{ and } n \text{ be an arbitrary value}$$

<figure>
  <picture>
    <source srcset="/images/ANN_with_labels_dark_mode.png"
            media="(prefers-color-scheme: dark)">
    <img src="/images/ANN_with_labels_light_mode.png"
         style="width:60%; display:block; margin:auto;"
         alt="A typical Neural Net">
  </picture>
  <figcaption style="text-align:center;">A typical Neural Net</figcaption>
</figure>





$$W_{h_1} \leftarrow W_{h_1} - \alpha \left( \frac{\partial o_1}{\partial W_{h_1}} + \dots + \frac{\partial o_n}{\partial W_{h_1}} \right)$$ $$b_{h_1} \leftarrow b_{h_1} - \alpha \left( \frac{\partial o_1}{\partial b_{h_1}} + \dots + \frac{\partial o_n}{\partial b_{h_1}} \right)$$

# Deriving gradients in the vector case
<figure data-latex-placement="h">
  <picture style="display:block; margin:auto; width:60%;">
    <!-- Dark mode image -->
    <source srcset="/images/Deriving_Backpropagation_Gradients_imgs/dark_mode.png" media="(prefers-color-scheme: dark)">
    <!-- Light mode fallback -->
    <img src="/images/ANN_with_labels_light_mode.png" alt="A typical Neural Net" style="width:100%;" />
  </picture>
  <figcaption style="text-align:center;">A typical Neural Net</figcaption>
</figure>




# Deriving gradients in earlier layers

We are trying to find $\frac{\partial o}{\partial W_i}$. Through the chain rule, we can expand this into:

$$\frac{\partial o}{\partial W_i} = \frac{\partial o}{\partial W_h}\frac{\partial W_h}{\partial W_i}$$