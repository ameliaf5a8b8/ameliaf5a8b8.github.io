---
title: "Planning and learning"
date: 2026-03-24
lastmod: 2026-03-24
tags: 
categories: ["Machine Learning", "Reinforcement Learning",Action-Value]
math: true
summary:
---

<div class="info-box">
This is a long term WIP. Information may not be accurate.
</div>

# State-space and Plan-space planning

## Plan-space planning
In plan-space planning, planning is a search through the space of plans.

Examples of such methods include
- policy gradient
- MC control, SARSA, and variants

## State-space planning
All state-space planning methods involve
- computing value functions as a key intermediate step toward improving the policy
- through updates or backup operations[^1] applied to simulated experience.

Example of a such a method is Dynamic Programming.
![[state-space-planning_diagram.png]]


reading section 8.2, pg 161

# Tabular DynaQ

<div style="display: flex; justify-content: space-between; align-items: flex-start;">

<div style="width: 65%;">
    Dyna-Q is a state-space planning approach that combines both model-free and model-based methods. It beings with the model free component by performing direct RL updates through Q-learning on real experience samples. Based on these samples, we estimate the model, which may not be defined for all states. We then perform a planning update through Q-learning by generating and learning simulated experiences.
  </div>
<figure style="width:45%">
  <img 
    src="general_Dyna_architecture.png"
    alt="General Dyna structure"
    style="display:block; margin: 0 auto;"
  >
  <figcaption style="text-align:center;">
    <strong>Figure 1:</strong> General Dyna structure
  </figcaption>
</figure>
</div>

**Pseudocode for DynaQ from Sutton Barto**   
Initialise $Q(s, a)$ and $\textit{Model}(s, a)$ for all $s\in \mathcal S$ and $a \in \mathcal A$   
Loop forever:   
$\hspace{2em}$(a) $S \leftarrow$ current (nonterminal) state   
$\hspace{2em}$(b) $A \leftarrow \varepsilon\text{-greedy}(S, Q)$   
$\hspace{2em}$(c) Take action $A$; observe resultant reward, $R$, and state, $S'$   
$\hspace{2em}$(d) $Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma \max_{a} Q(S', a) - Q(S, A) \right]$   
$\hspace{2em}$(e) $Model(S, A) \leftarrow R, S'$ (assuming deterministic environment)   
$\hspace{2em}$(f) Loop repeat $n$ times:   
$\hspace{4em}$$S \leftarrow$ random previously observed state   
$\hspace{4em}$$A \leftarrow$ random action previously taken in $S$   
$\hspace{4em}$$R, S' \leftarrow Model(S, A)$   
$\hspace{4em}$$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma \max_{a} Q(S', a) - Q(S, A) \right]$     
## Modeling errors

Models may be incorrect because the environment is stochastic and only a limited number of samples have been observed, or because the model was learned using function approximation that has generalised imperfectly, or simply because the environment has changed and its new
behaviour has not yet been observed.

In some cases, the suboptimal policy computed by planning quickly leads to the
discovery and correction of the modeling error. This tends to happen when the model
is optimistic in the sense of predicting greater reward or better state transitions than
are actually possible. The planned policy attempts to exploit these opportunities and in
doing so discovers that they do not exist.

## DynaQ+



[^1]: a backup operation is just an RL update
