---
title: "Upper Confidence Bound"
date: 2026-03-07
lastmod: 2026-03-08
tags: []
categories: []
math: true
summary:
---
# Introduction

The commonly used epsilon-greedy methods has a significant drawback --- it explores a random action with no consideration of its likelihood to be the optimal actions. It would be better to selectively explore actions according to their potential to be optimal, based on their current action value and its associated uncertainties.

# UCB

UCB attempts to mitigate this issue through the addition of an uncertainty term, to form a *maximising* action. The action selection becomes $$A_t \doteq \underset{a}{\text{argmax}} \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$ where $Q_t(a)$ denotes the action value at time step $t$, $N_t(a)$ denotes the number of times action $a$ has been selected prior to $t$, and $c >0$ controls the degree of exploration. If $N_t(a) = 0$, then $a$ is considered to be a maximising action.

The idea of UCB is that the square root term represents the uncertainty in the action-value estimate. The quantity being argmaxed over becomes sort of upper bound on $Q(a)$ [^1], with $c$ determining the confidence level. Each time $a$ is selected, the uncertainty reduces. On the other hand, each time an action other than $a$ is selected, $\ln t$ increases, increasing the uncertainty. With the logarithm, increases in uncertainty get smaller over time, while remaining unbounded, causing all actions to be selected in future time steps, regardless of the current time step.


<figure id="fig:1">
  <picture>
    <source srcset="../ucb_reward_dark.svg"
            media="(prefers-color-scheme: dark)">
    <img src="../ucb_reward_light.svg"
         style="width:100%; display:block; margin:auto;"
         alt="The effect of UCB on the 10-armed bandit problem, averaged over 5000 runs. Both methods used a constant step size \alpha = 0.1.">
  </picture>
  <figcaption style="text-align:center;">
    <strong>Figure 1:</strong> The effect of UCB on the 10-armed bandit problem, averaged over 5000 runs. Both methods used a constant step size \alpha = 0.1.  </figcaption>
</figure>

# Exercise

#### Initial spike.

The results in Figure 1 should be quite reliable because they are averaged over 5000 individual, independent 10-armed bandit tasks. Explain the oscillations in the early part of the UCB curve, particularly at the 11th time step where the average reward jumps to $1.1$. Note that for your explanation to be complete, it must explain why the rewards increases on the 11th step and decreases on the subsequent steps

<figure id="fig:2">
  <picture>
    <source srcset="../Avg_reward_zoomed_in_dark.svg"
            media="(prefers-color-scheme: dark)">
    <img src="../Avg_reward_zoomed_in_light.svg"
         style="width:100%; display:block; margin:auto;"
         alt="The effect of UCB on the 10-armed bandit problem, averaged over 5000 runs. Both methods used a constant step size \alpha = 0.1.">
  </picture>
  <figcaption style="text-align:center;">
    <strong>Figure 2:</strong> The effect of UCB on the 10-armed bandit problem, averaged over 5000 runs. Both methods used a constant step size \alpha = 0.1.  </figcaption>
</figure>

#### Observation.

By examining the nature in which UCB performs action selection, hypothesise how $\varepsilon$-greedy method has a higher % Optimal action than UCB but lower average reward. [^2]

<figure data-latex-placement="H">
<figure id="">
  <picture>
    <source srcset="../ucb_reward_c_2_dark.svg"
            media="(prefers-color-scheme: dark)">
    <img src="../ucb_reward_c_2_light.svg"
         style="width:100%; display:block; margin:auto;"
         alt="">
  </picture>
  <figcaption style="text-align:center;">
    <strong>Figure 3:</strong>   </figcaption>
</figure>

<figure id="">
  <picture>
    <source srcset="../ucb_optimal_action_c_2_dark.svg"
            media="(prefers-color-scheme: dark)">
    <img src="../ucb_optimal_action_c_2_light.svg"
         style="width:100%; display:block; margin:auto;"
         alt="">
  </picture>
  <figcaption style="text-align:center;">
    <strong>Figure 4:</strong>   </figcaption>
</figure>

<figcaption>UCB versus <span class="math inline"><em>ε</em></span>-greedy on different axes, averaged over 50 000 runs. Both methods used a constant step size <span class="math inline"><em>α</em> = 0.1</span></figcaption>
</figure>

# Conclusion

UCB is an excellent way to handle the explore-exploit trade-off. However, it is much more difficult than $\varepsilon$-greedy methods to extend beyond bandits into the general non-stationary case. Another difficulty arises when dealing with large state spaces, particularly when dealing with function approximation. For these reasons, UCB is a powerful tool, but is typically not practical in more advanced settings.

# Appendix

## Answers to Exercise questions

#### Initial spike.

At time step 10, the agent has tried each action exactly once. Hence, the uncertainty term on all actions are equal, which allows the agent to choose the action with the highest action value at time step 11. After time step 11, the uncertainty term on the high action value decrease, prompting the agent to choose actions with lower action values. This phenomenon repeats, prompting the agent to choose worse actions each time, until the $N_t(a)$ term becomes larger and the changes in uncertainty term become less extreme.

#### Observation. 

Even though the $\varepsilon$-greedy method has a higher chance of choosing the optimal action. UCB's exploration is targeted toward actions that are near optimal, which provides high rewards. Averaged out, UCB's exploration is less costly to the agent than the random exploration in the $\varepsilon$-greedy method.

Note that in the long run, UCB tends to outperform the $\varepsilon$-greedy method

<figure data-latex-placement="H">
<figure id="">
  <picture>
    <source srcset="../ucb_reward_c_2_long_term_dark.svg"
            media="(prefers-color-scheme: dark)">
    <img src="../ucb_reward_c_2_long_term_light.svg"
         style="width:100%; display:block; margin:auto;"
         alt="">
  </picture>
  <figcaption style="text-align:center;">
    <strong>Figure 5:</strong>   </figcaption>
</figure>

<figure id="">
  <picture>
    <source srcset="../ucb_optimal_action_c_2_long_term_dark.svg"
            media="(prefers-color-scheme: dark)">
    <img src="../ucb_optimal_action_c_2_long_term_light.svg"
         style="width:100%; display:block; margin:auto;"
         alt="">
  </picture>
  <figcaption style="text-align:center;">
    <strong>Figure 6:</strong>   </figcaption>
</figure>

<figcaption>UCB performs better than <span class="math inline"><em>ε</em></span>-greedy on both axes in the long run. The data is averaged over 50 000 runs, with a constant step size <span class="math inline"><em>α</em> = 0.1</span></figcaption>
</figure>

## Simulation code



[^1]: Technically, the upper bound of an action-value estimate is always $\infty$. However, by adjusting $c$, we can ensure that the true mean $q_*$ is less than or equal to our upper bound with probability $P = 1- t^{-2c}$

[^2]: UCB tends to perform better than $\varepsilon$-greedy methods, both in terms of average reward and % Optimal action. Make your hypothesis based on the provided data.

