---
title: "Finite MDPs"
date: 2026-03-12
lastmod: 2026-03-12
tags: [Easy]
categories: ["Machine Learning", "Reinforcement Learning",Action-Value]  
draft: true
math: true
summary:
---

<!-- NOTE:draft is set to true. it will not be shown , but still accessible by link -->

# Characteristics of MDPs


In a *finite* MDP, the set of states, actions, and rewards ($\mathcal S$, $\mathcal A$, and $\mathcal R$) all have a finite number of elements. For particular values of $S_t$ and $A_t$, there is a joint probability that the next state and reward will be $s^\prime$ and $r$, respectively.
$$\boxed{p(s^\prime,r \mid s,a) \doteq P(S_{t+1} = s^\prime, R_{t+1} = r, \mid  S_{t}=s, A_{t}= a)}$$

where the probabilities given by $p$ completely characterise the *dynamics* of the environment.

A state representation satisfies the Markov property if the probability of each possible value of $S_{t+1}$ and $R_{t+1}$ must depend only on the current state and action, $S_{t}$ and $A_{t}$, and completely independent on all earlier states and actions. This can be interpreted as a restriction on the state to include all information required to determine the probability distribution of the next state and reward, given the current action. In most Reinforcement Learning tasks, the state is assumed to hold the Markov property.

# Returns and episodes
The goal of the agent is to maximise the *expected return*, denoted $G_t$, is defined as some specific function of future rewards. In the simplest case, the return is is the sum of future rewards.
<span id="eqn:1"></span>
$$G_t \doteq  R_{t+1} +  R_{t+2} +  R_{t+3} +\cdots + R_T \tag 1$$
where $T$ is the final time step. This approach makes sense when there is a natural notion of a final time step, in which the sequence of agent–environment interactions naturally terminates. This sequence of interactions is called *episodes*.

However, not all tasks naturally terminate and can be broken down into discrete episodes. We call these *continuing* tasks. In these cases, the return formulation in [(1)]("#eqn:1") is problematic as the return, which we are trying to maximise, could be unbounded.

To address this issue, we introduce the concept of discounting.
$$\begin{align}G_t &\doteq  R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} \notag\\
&= \sum_{k=0}^\infty \gamma^k R_{t+k+1}\end{align}$$ <span id="eqn:2"></span>
where $\gamma \in [0,1]$, a hyper parameter, is the *discount rate*.
The discount rate determines the present value of future rewards. If $\gamma < 1$, the infinite sum in [(2)](#eqn:2) has a finite value as long as the reward sequence ${R_k}$ is bounded. 