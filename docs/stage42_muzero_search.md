# Stage 4.2 — MuZero MCTS / PUCT search

Implementation date: 2026-09-10. This stage adds the **search layer** only:
given one real Forex observation, it builds a latent tree with MuZero
`initial_inference` / `recurrent_inference` and returns an improved root policy
over the frozen six-action space.

Still **not** implemented: MuZero training loss, replay buffer, trajectory
storage, self-play orchestration, reanalysis, distributed MCTS, prioritized or
decision-rich replay, n-step target generation, Stochastic MuZero.

The success criterion is not profitability. It is that search plans correctly
in latent space, respects deterministic position/action constraints, backs up
discounted single-agent returns correctly, and returns a valid improved policy.
The network has still not learned a Forex world model.

---

## 1. Files added / changed

Added:

| File | Purpose |
|---|---|
| `forexmind/muzero/actions.py` | Six-action MuZero space, env-mask projection, `PlanningState`. |
| `forexmind/muzero/minmax.py` | `MinMaxStats` value normalization. |
| `forexmind/muzero/node.py` | `Node` tree abstraction with lazy expansion. |
| `forexmind/muzero/search.py` | `MuZeroMCTS`, `SearchConfig` re-export, `SearchResult`, `discounted_backup`, `visit_count_policy`. |
| `tools/benchmark_muzero_search.py` | Search-cost benchmark (16/32/64/128 simulations). |
| `tests/test_muzero_actions.py` | Six-action space, projection, planning state (36 tests). |
| `tests/test_muzero_search.py` | MCTS/PUCT, backup, minmax, noise, masks, reproducibility (43 tests). |
| `docs/stage42_muzero_search.md` | This report. |

Changed:

| File | Change |
|---|---|
| `forexmind/muzero/config.py` | `num_actions` default → 6; new `SearchConfig`. |
| `forexmind/muzero/__init__.py` | Exports the search API. |
| `tests/test_muzero_networks.py` | Retargeted to six actions and valid indices. |
| `docs/stage41_muzero_networks.md` | Updated for the six-action model. |
| `README.md` | Stage 4.2 section + benchmark command. |

Unchanged: the shared ten-action environment, PPO, the evaluator, configs, and
all Stage 3 / Stage 3.5 behaviour. `data/reports/stage41_muzero_inference_benchmark.json`
and `data/reports/stage42_muzero_search_benchmark.json` are regenerated.

## 2. MCTS architecture

```
real observation
      ↓  initial_inference(observation, action_mask)         (one call)
root latent state + root policy logits + root value
      ↓  expand root: priors = softmax(masked logits), one child per VALID action
      ↓  optional root Dirichlet noise over valid actions only
   ┌─ repeated simulations (num_simulations) ────────────────────────────┐
   │  selection : descend while node.is_expanded(), PUCT argmax          │
   │  evaluation: recurrent_inference(parent.latent, action)  (1 call)    │
   │  expansion : deterministic planning mask -> priors -> lazy children  │
   │  backup    : G_k = r_{k+1} + gamma * G_{k+1} up the visited path     │
   └──────────────────────────────────────────────────────────────────────┘
      ↓
visit counts -> pi(a) ∝ N(a)^(1/T), selected action, SearchResult
```

Properties enforced by construction:

* The module never references `env.step()`; every imagined transition goes
  through `model.recurrent_inference`. Only the root mask/planning state are
  supplied by the caller from the live environment.
* **Lazy expansion** — children are created (with priors) when a node is
  expanded, but `recurrent_inference` only runs for an edge when it is
  traversed for the first time.
* **Invalid actions have no child at all**, so they cannot be selected,
  expanded, or given visits.
* All neural calls run under `torch.no_grad()`; search never builds a graph.

Cost sanity (asserted in tests): with `N` simulations a search performs exactly
`N` recurrent calls, expands `N + 1` nodes (root + one leaf each), and root
child visits sum to `N`.

**Batch-ready seam.** All recurrent calls go through
`_recurrent_inference(latents, actions, stats)`, which already builds a batched
`[N, latent_dim]` tensor. The sequential search passes a single element per
call; a later version can evaluate several leaves (or several environments) with
one call without restructuring. No asynchronous batching is implemented.

## 3. PUCT equation used

```
U(s,a) = pb_c(s) * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
pb_c(s) = log((N(s) + pb_c_base + 1) / pb_c_base) + pb_c_init

Q(s,a) = r(s,a) + gamma * V(child)          # reward of the edge, on the child
value_score = minmax.normalize(Q(s,a))      # 0 when the child is unvisited

score(s,a) = value_score + U(s,a)
select      = argmax_a score(s,a)
```

* `P(s,a)` is the masked-softmax prior; invalid actions have `P = 0` and no
  child, so they can never win the argmax.
* `Q` is the **reward-inclusive** value of taking the action. The brief's
  shorthand "`Q(s,a) = child.value`" is implemented as
  `child.reward + discount * child.value()` because the reward of the transition
  is stored on the child node (see §5); without the reward term PUCT would
  ignore the immediate reward entirely.
* `pb_c_base`, `pb_c_init`, `num_simulations`, `discount`, and
  `normalize_values` are `SearchConfig` fields — no search constant is
  hard-coded in the implementation.
* Ties are resolved by the lowest action index (stable, deterministic).

## 4. Discount handling

`SearchConfig.discount` (default `0.99`) is used in exactly two places: the PUCT
`Q(s,a)` term and the backup recursion. The same value is intended for the
MuZero training loss later; the brief's requirement to use one consistent
discount is satisfied by having a single config field. `discount` must lie in
`(0, 1]` and is validated.

## 5. Single-agent backup behavior

```
G_K     = leaf_value
G_k     = r_{k+1} + gamma * G_{k+1}
```

Rewards are **added**, never sign-flipped, and there is no player alternation or
`max`/`min` switching — ForexMind is a single-agent trading environment. This is
implemented as a pure function `discounted_backup(rewards, leaf_value, discount)`
so it can be verified numerically.

`Node.reward` is the reward on the edge from the parent into that node (root
reward `0`), which is why the PUCT value term is reward-inclusive.

## 6. Min-max normalization

`MinMaxStats` tracks `[minimum, maximum]` over every value backed up during a
single search. `normalize(value)` returns `0.0` when uninitialized, when the
spread is `<= 1e-8` (single distinct value), and is always finite. Values are
**not** clamped to `[0, 1]`: a value outside the observed range is a legitimate
signal and MuZero's reference implementation does not clamp.

One `MinMaxStats` instance is created per search, so searches are independent.

## 7. Root Dirichlet noise

```
P'(a) = (1 - f) * P(a) + f * noise(a),   noise ~ Dirichlet(alpha)
```

* `root_dirichlet_alpha` (0.3) and `root_exploration_fraction` (0.25) are
  `SearchConfig` fields.
* Applied **only over valid root actions**; masked actions keep `P = 0`.
* Off by default (`SearchConfig.add_root_noise = False`). `SearchConfig.training()`
  enables it, `SearchConfig.evaluation()` disables it; `search(add_root_noise=...)`
  can override per call.
* The RNG is a `numpy.random.Generator` seeded from `SearchConfig.seed`, so noise
  is reproducible and can be injected by the caller. Deterministic evaluation
  never consumes it.

## 8. Six-action masking behavior

MuZero's action space is `(HOLD, FLAT, SHORT_100, SHORT_50, LONG_50, LONG_100)`;
`MUZERO_ENV_ACTION_INDICES = (0, 1, 2, 4, 7, 9)` maps it onto the shared
ten-action environment, and both names and target exposures are *derived* from
`forexmind.environment.actions` (so they cannot drift). ±25/±75 are not
reachable.

`project_action_mask` accepts a ten-wide environment mask (normal case) or an
already-projected six-wide mask. Reductions with no MuZero equivalent (e.g.
`SHORT_75`) correctly mask nothing.

Root mask example with the real environment at `LONG_50`:

| action | HOLD | FLAT | SHORT_100 | SHORT_50 | LONG_50 | LONG_100 |
|---|---|---|---|---|---|---|
| valid | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |

`HOLD` is always valid, and a test asserts this for every planning state.

## 9. Deterministic planning-position state

The neural latent cannot expose future action validity, and the brief forbids
faking future masks from unavailable information. Search therefore carries a
deterministic `PlanningState(exposure, is_flat)` next to the neural latent:

| MuZero action | planning transition |
|---|---|
| HOLD | unchanged |
| FLAT | `(0.0, is_flat=True)` |
| SHORT_100 | `(-1.0, False)` |
| SHORT_50 | `(-0.5, False)` |
| LONG_50 | `(+0.5, False)` |
| LONG_100 | `(+1.0, False)` |

`PlanningState.action_mask()` calls the environment's own
`valid_action_mask(exposure, is_flat)` and projects it, so the rule is *reused*,
never re-implemented. `PlanningState.from_env(env)` mirrors
`ForexEnvironment.action_masks()` exactly — tested for parity across eight
action sequences. `PlanningState.from_action_mask(mask)` reconstructs the state
from a six-wide mask; this is faithful for every state reachable by MuZero
actions because only `is_flat` and "is the exposure already a MuZero target"
affect the six-wide mask.

`recurrent_inference` returns only `(reward, next latent)`; the planning state is
updated separately and deterministically. The mechanism is intentionally generic
and documented.

## 10. Visit-count policy construction

```
pi(a) ∝ N(root, a) ** (1 / temperature)
```

* `temperature <= 1e-8` → deterministic argmax over visit counts (evaluation).
* `temperature > 0` → normalized `N^(1/T)` over visited actions, and the
  returned action is sampled from `pi` with the search RNG.
* The returned `policy` is six-wide and sums to exactly 1; invalid actions have
  probability 0.

`SearchResult` carries `action`, `visit_counts`, `policy`, `root_value`,
`root_priors`, `root_action_mask`, and `diagnostics` (simulations, tree depth,
expanded nodes, recurrent calls, root predicted value, per-action `Q`, PUCT
scores, and predicted immediate rewards). Together with the observation the
caller already holds, this is everything a future replay example needs; the real
environment reward is only known after the environment is stepped, and no replay
buffer is built here.

## 11. Toy-tree tests

A table-driven `ToyModel` (exact policy/value/reward/transition lookups)
implements the Stage 4.1 API, so search behaviour is asserted exactly rather than
statistically.

* Actions with rewards `[0, -0.1, -0.3, -0.05, +0.2, +0.05]` → visits concentrate
  on `LONG_50` (`argmax(visit_counts) == LONG_50`, `action == LONG_50`).
* Root visits sum to the simulation budget; `recurrent_inference` calls ==
  simulations; expanded nodes == simulations + 1.

## 12. Multi-step planning test

```
SHORT_100: immediate +1, lands on a state with value -10  → Q ≈ -8.9
LONG_50:   immediate  0, lands on a state with value  +5  → Q ≈ +4.95
others:    immediate -5, lands on a state with value -10  → Q ≈ -14.9
```

The greedy choice (`SHORT_100`, highest immediate reward) is **not** selected:
with 64 simulations search selects `LONG_50`, gives it more visits than
`SHORT_100`, and its backed-up `Q` reflects discounted future value
(`4.5 < Q(LONG_50) <= 5.0`, `-10 < Q(SHORT_100) < -8`). Search is planning, not
greedy reward maximisation.

## 13. HOLD / FLAT tests

Three toy trees whose absorbing best state is reached only by `HOLD`, only by
`FLAT`, and only by `LONG_50`: search selects each of them in turn. A fourth tree
makes `HOLD` the worst action and search still explores it but does not select
it, and the uniform-prior test confirms `P(HOLD) = 1/6` with no rescaling.

`test_search_has_no_action_specific_special_casing` also asserts the search
module defines no `HOLD`/`FLAT`/`SHORT_*`/`LONG_*` constant, so there is no
`HOLD prior *= 0.5`, no negative HOLD bonus, and no HOLD down-sampling. All
exploration comes from PUCT, Dirichlet noise, visit counts, and temperature.

## 14. Mask tests

* Root: a masked action receives prior `0.0`, `0` visits, `pi = 0`, and is never
  selected — even when it would be overwhelmingly the best action.
* Imagined nodes: every expanded node's children exactly match its deterministic
  planning mask, and no node with a flat planning state has a `FLAT` child.
* `HOLD` remains valid in every planning state.
* A supplied `planning_state` inconsistent with the given mask is rejected.
* Real-network search with a masked root action gives it `0` visits.

## 15. Backup tests

* `discounted_backup([1, 2], 3.0, 0.9) == [5.23, 4.7, 3.0]` exactly
  (`G2 = 3`, `G1 = 2 + 0.9*3`, `G0 = 1 + 0.9*G1`).
* Positive and negative rewards are added with one sign convention
  (`+1 → +1`, `-1 → -1`); no sign flipping, no player alternation.
* No-transition path returns the leaf value; invalid discount is rejected.

## 16. Search reproducibility

With the same model, observation, mask, seed, and config, two independent
`MuZeroMCTS` instances return identical visit counts, policy, and selected
action. Repeated evaluation searches on one instance are also identical
(root noise disabled, `temperature = 0`, no sampling anywhere in the network).
With noise enabled and a fixed seed, root priors and visits are reproducible
too.

## 17. Search throughput

`python -m tools.benchmark_muzero_search --simulations 16 32 64 128 --repeats 25`
on CPU (`AMD64`, `torch 2.13.0+cpu`, latent 128, hidden 256×2, obs 351, 6
actions, one fixed root observation, evaluation config):

| Simulations | searches/s | sims/s | ms/search | recurrent calls/search | nodes/search | depth |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 17.4 | 278 | 57.5 | 16.0 | 17.0 | 5.00 |
| 32 | 9.1 | 292 | 109.6 | 32.0 | 33.0 | 7.00 |
| 64 | 4.5 | 290 | 220.4 | 64.0 | 65.0 | 8.00 |
| 128 | 2.2 | 288 | 445.0 | 128.0 | 129.0 | 9.00 |

Search cost is linear in simulations at roughly **290 simulations/s** (~2.9×
the measured Stage 4.1 `recurrent_inference` rate at batch 1, because a search
does 1 initial + N recurrent calls plus Python-side overhead). Depth is the
final tree depth and is identical across repeats because evaluation search is
deterministic. Results are written to
`data/reports/stage42_muzero_search_benchmark.json`; no optimisation was
attempted. Run-to-run spread on this shared machine is significant, so treat
these as an order of magnitude.

## 18. Search cost at 16 / 32 / 64 / 128 simulations

| Simulations | ms/search | recurrent calls | expanded nodes | depth |
|---:|---:|---:|---:|---:|
| 16 | 57.5 | 16 | 17 | 5 |
| 32 | 109.6 | 32 | 33 | 7 |
| 64 | 220.4 | 64 | 65 | 8 |
| 128 | 445.0 | 128 | 129 | 9 |

The default budget is `SearchConfig.num_simulations = 50` — deliberately modest,
because quality-versus-cost must be measured before spending hundreds of
simulations per decision.

## 19. Remaining concerns

- **No terminal modelling.** The learned dynamics never predicts a terminal
  state, so search can plan past the environment's episode horizon / liquidation
  boundary. Episodes are bounded by the environment, not by search; a value
  bootstrap at truncation belongs to the training stage.
- **Untrained network.** Value/reward heads are random, so search quality on the
  real network is meaningless until training. All behavioural claims here use
  the table-driven toy model.
- **Planning state granularity.** Only the discrete exposure state is tracked
  deterministically; margin, free margin, and forced-liquidation states are not.
  `PlanningState` is the documented extension point.
- **Single-leaf inference.** Search evaluates one leaf per simulation. The
  batched seam exists but is unused, so the benchmark is a lower bound on what a
  batched implementation would achieve.
- **Depth grows with budget.** Depth reached 9 at 128 simulations with no depth
  cap; very large budgets could produce long, thin trees before breadth fills in.
- **Root policy is not the action distribution for training yet.** Sampling at
  nonzero temperature is implemented, but its interaction with the replay
  targets is a Stage 4.3 decision.
- CUDA remains unavailable on this machine, so device parity is covered by a
  skipped test.

## 20. Readiness for Stage 4.3 trajectory collection and replay

Yes. Search already produces exactly what trajectory collection needs:

- `SearchResult.action`, `.policy`, `.visit_counts`, `.root_value`,
  `.root_priors`, `.root_action_mask`, plus per-action `Q` and predicted rewards
  in `.diagnostics`;
- the caller holds the observation and the real environment reward/mask, so a
  replay example `(observation, mask, policy, action, root_value)` is
  constructible without touching search internals;
- `search_from_env(env, observation)` reads the live mask and planning state,
  and the mask is validated against the planning state so drift is caught;
- search is deterministic in evaluation mode and reproducible with noise, which
  matters for replay integrity;
- all neural work is `no_grad` and batched-shaped, so a collector can later call
  it at volume.

Deliberately absent: replay buffer, trajectory storage, self-play workers,
training loss, target construction, reanalysis, and decision-rich replay
balancing.
