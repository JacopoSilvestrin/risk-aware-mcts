# Handout: MCTS bug fixes ported from `RiskAwareMCTS-ROBOT`

This branch (`bugfixes-from-robot`) brings the algorithmic bug fixes made in the later
`RiskAwareMCTS-ROBOT` project back into the original PPSantos code base. Only the fixes
and the minimal caller edits were ported; the robot / bin-packing / Gemini-planner work
was left out on purpose.

Notation: `C` is the (discounted) cost, `beta` is `erm_beta`, and the entropic risk measure is
`ERM_beta(C) = (1/beta) * log E[exp(beta * C)]`.

## Summary

| # | File | Fix | Type | Changes results? |
|---|------|-----|------|------------------|
| 1 | `algos/mcts.py` | Sign of the tree-edge reward in `grow_tree()` | Real bug | Yes |
| 2 | `algos/mcts.py` | UCB scoring in ERM cost units (`_estimate_erm`) | Design fix | Yes |
| 3 | `algos/mcts.py` | `best_action()` picks lowest ERM, not most visited | Design fix | Yes |
| 4 | `algos/erm_mcts.py` | `root_depth` constructor argument was ignored | Real bug | Yes |
| 5 | `algos/erm_mcts.py` | `best_action()` picks lowest ERM (`min_erm`), with legacy flag | Design fix | Yes |
| 6 | `algos/erm_mcts.py` | Log-sum-exp ERM estimator (`_estimate_erm`) | Numerical | No |
| 7 | `algos/erm_backward_induction.py` | Log-sum-exp Q-values | Numerical | No |
| 8 | `simulate_mcts_accrued_costs.py` | Clip `exp(beta * cost)` (`_safe_exp`) | Numerical | Only at large beta |

"Real bug" means the old code did something unintended. "Design fix" means the old code did what
it said, but the choice was statistically unsound for a risk-sensitive objective.

---

## 1. Sign bug in acc-mcts tree edges (`algos/mcts.py`, `grow_tree`)

**Background.** `MCTS` is a reward-*maximising* MCTS. It is driven through the accrued-cost wrapper
(`AccruedCosts_MDP`), whose `step()` returns a *cost*: `0` at every intermediate step and
`exp(beta * accrued_cost)` at the terminal step. `rollout()` converts cost to reward correctly
(`return (-1.0) * R`).

**Bug.** `grow_tree()` stored the edge value as `random_node.reward = r`, without negating it.
A backup that reached the terminal transition through an already-built tree edge therefore added
`+exp(beta * C)`, while a backup that reached it through a rollout added `-exp(beta * C)`.
The running mean used by UCB and `best_action()` mixed the two signs.

**Fix.** `random_node.reward = -r`.

**Symptom / evidence.** On `four_state_mdp` (H=8, beta=0.5, 500 iterations, seed 1) the *original*
code produced 30 of 562 random nodes with a positive `cumulative_reward`, which is impossible
when every reward is `-exp(...) <= 0`. After the fix there are 0 of 996.

## 2. UCB scoring in cost units (`algos/mcts.py`, `select` + new `_estimate_erm`)

**Problem.** UCB compared `mean_reward + K * sqrt(sqrt(N) / n)`. Here `mean_reward` is
`-mean(exp(beta * C))`, which is astronomically large after even one bad sample (for example an
accrued cost of 200 at beta=1 gives `exp(200) ~ 1e87`). The exploration bonus is of order 1-10, so it
can never compete: the first catastrophic sample effectively removes that branch from the search
forever, however much planning budget remains.

**Fix.** Read the mean reward back out in cost units:

```python
_estimate_erm(node) = (1 / beta) * log(-cumulative_reward / visits)
```

This is valid because every visited node's `cumulative_reward / visits` equals
`-mean(exp(beta * C))`. By Jensen's inequality the result lies between `E[C]` and `max(C)`, so
it is commensurate with the exploration bonus. `select()` now scores
`-_estimate_erm(child) + K * sqrt(sqrt(N) / n)`.

**Consequence: `MCTS.__init__` now requires `erm_beta`**
(`MCTS(initial_state, env, K_ucb, erm_beta, rollout_policy=None)`). It must match the `erm_beta`
the wrapper env uses to build the terminal reward.

## 3. `MCTS.best_action()`: lowest ERM instead of most visited (`algos/mcts.py`)

Most-visited is a valid final-decision rule for expected-value MCTS, because UCB1 concentrates visits
on the best-mean arm. ERM is dominated by tail behaviour and the exploration bonus deliberately keeps
sampling arms whose tail estimate is still noisy, so visit counts do not track the ERM ranking.
`best_action()` now returns the root action with the lowest `_estimate_erm` (unvisited children count as
`inf`).

## 4. `ERMMCTS` ignored `root_depth` (`algos/erm_mcts.py`, `__init__`)

**Bug.** The constructor accepted `root_depth` and then ran `self.root_depth = 0`.

**Why it matters.** `select()` uses `beta_depth = erm_beta * gamma**depth`. The intended depth is
`root_depth + depth_in_tree`. Every caller that *rebuilds* a tree mid-episode
(`ERMMCTS(..., root_depth=t+1)`, the `else` branch after `update_root_node` fails) silently got
`root_depth = 0`. Because `gamma < 1`, `beta_depth` was too large, so rebuilt trees were more
risk-averse than the decay schedule intends, and increasingly so later in the episode. Trees that were
*reused* (`set_root_depth`) were fine, so the bug was intermittent.

**Fix.** `self.root_depth = root_depth`.

## 5. `ERMMCTS.best_action()`: `min_erm` default (`algos/erm_mcts.py`)

Same reasoning as fix 3. `best_action()` used to return the most visited root child. There are now two
new constructor arguments:

* `best_action_criterion="min_erm"` (default) returns the lowest empirical ERM, evaluated at
  `root_depth`. `"most_visited"` reproduces the legacy behaviour so old results can be reproduced.
* `risk_neutral_beta_threshold=1e-6`. When `erm_beta <=` this value the method uses `most_visited`
  regardless. At near-zero beta, `ERM ~ E[C]`, so `min_erm` reduces to picking the lowest raw mean,
  which is easily fooled by an under-sampled but lucky branch; visit counts are more robust there. Pass
  `None` to disable the override.

To reproduce pre-fix numbers: `ERMMCTS(..., best_action_criterion="most_visited")`. (Note that fix 4
also affects old numbers, and has no flag.)

## 6. Log-sum-exp ERM estimator (`algos/erm_mcts.py`)

`select()` computed `(1/b) * log(mean(exp(b * costs)))` inline, which overflows once `b * cost` exceeds
about 709. It now calls `_estimate_erm(costs, depth)`, which subtracts `max(b * costs)` before
exponentiating. The result is mathematically identical and never overflows. `best_action()` reuses it.

## 7. Log-sum-exp in backward induction (`algos/erm_backward_induction.py`)

Same trick for `Q = (1/b) * log(P . exp(b * (C + gamma * V)))`: subtract `max(x)` from the exponent
vector before `np.exp` and add it back after the `log`. Identical results, no overflow at large beta or
long horizons.

## 8. Overflow guard for the terminal reward (`simulate_mcts_accrued_costs.py`)

`AccruedCosts_MDP.step()` returned `np.exp(beta * accrued_cost)`, which becomes `inf` above about 709.78
(float64). One `inf` makes `RandomNode.cumulative_reward` `-inf` forever, so all UCB scores tie and the
first child always wins. The exponent is now clipped at `MAX_EXP_ARG = 700` via `_safe_exp`. The `MCTS(...)`
calls in this file also pass `erm_beta` (fix 2).

---

## Intentionally NOT ported

* The state-hash change for the POMDP bin-packing environment (`state["bags"]` in `ERMMCTS._hash_state`).
  That environment is not part of this project.
* Everything robot-specific: bin-packing envs, Gemini / rule-based planners, `standard_mcts.py`,
  `bandit_env.py`, `long_grid` MDPs, the `debug_*`, `diagnose_*` and `verify_*` scripts, notebooks,
  `.env`, extra `requirements.txt` entries.
* ROBOT's experiment configs (`N=100`, `env="binpacking"`, ...). The defaults in this repo's `simulate_*.py`
  are untouched.
* Cosmetic edits (moving `select_outcome`, a dead `if ... and False: print(...)` debug block).

## Known limitations

* **acc-mcts state hashing.** `MCTS._hash_state` uses `(state, accrued_costs, t)`. The ROBOT investigation
  suspects the same class of hash-collision problem there for environments with hidden state, but it was not
  fixed even in ROBOT, so it is not fixed here. It is harmless for fully observed MDPs.
* **`np.log(-mean_reward)`** in `MCTS._estimate_erm` requires a visited node to have a strictly negative mean
  reward. That holds for the accrued-cost wrapper (every visit ends in a `-exp(...) < 0` terminal reward). A
  different env wrapper that returns 0 or positive terminal rewards would break it.
* **Results will change.** Fixes 1-5 alter search behaviour, so any numbers produced with the previous code
  are not directly comparable with new runs.

## How it was checked

Scratch script (not in the repo), on `four_state_mdp` / `two_paths_mdp`, with `RuntimeWarning` turned into errors:

* Backward induction: stabilised policy equals the naive-formula policy; beta=50, H=100 runs without overflow.
* `ERMMCTS`: `root_depth=7` is honoured; `_estimate_erm` equals the naive formula and survives costs of 5000;
  beta <= 1e-6 falls back to most-visited.
* `MCTS`: no random node has a positive `cumulative_reward` after the fix (the original code did); at beta=10
  all `cumulative_reward` values stay finite; `_safe_exp(3000)` is finite.
* `simulate_erm_mcts.simulate_ERM_MCTS` and `simulate_mcts_accrued_costs.simulate_accrued_MCTS` run end to end.

This is a smoke and consistency check, not a statistical re-validation of the algorithms. To compare
old and new behaviour on a metric that matters to you, re-run your experiments on both the `master` and
`bugfixes-from-robot` branches.

## Where the changes are

`git log master..bugfixes-from-robot` lists one commit per fix group; `git diff master..bugfixes-from-robot`
shows everything.
