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

# State-space and Plan-space planning.

## Plan-space planning
In plan-space planning, planning is a search through the space of plans.

Examples of such methods include
- policy gradient
- MC control, SARSA, and variants

## State-space plannig
All state-space planning methods involve
- computing value functions as a key intermediate step toward improving the policy
- through updates or backup operations[^1] applied to simulated experience.

Example of a such a method is Dynamic Programming.
![[state-space-planning_diagram.png]]


reading section 8.2, pg 161

[^1]: a backup operation is just an RL update
