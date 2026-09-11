# Stage 4.3 — trajectory collection, replay storage, and target construction

Implementation date: 2026-09-10. This stage builds the data-generation layer
between the real Forex environment, the Stage 4.1 MuZero networks, and Stage 4.2
MCTS::

    real observation -> MuZero initial inference -> MCTS -> selected real action
        -> env.step(action) -> real reward / next observation -> trajectory
        -> MuZero training targets

Still **not** implemented: gradient optimization, the MuZero loss, optimizer,
target network, reanalysis, prioritized replay, distributed actors,
decision-rich oversampling by default, a full training run, Stochastic MuZero.
The network has not been trained; the success criterion is that real
interactions are stored with correct indexing and can produce mathematically
correct policy/value/reward targets.

The frozen Forex reward `r_t = log(equity[t+1] / equity[t])`, execution timing,
accounting, currency conversion and the ten-action environment are all
untouched, and MuZero still uses its own six-action space.

---

## 1. Files added / changed

Added:

| File | Purpose |
|---|---|
| `forexmind/muzero/trajectory.py` | `MuZeroTrajectory`, `TrajectoryMetadata`, `model_version`. |
| `forexmind/muzero/targets.py` | `TargetConfig`, `value_target`, `MuZeroSample`, `MuZeroBatch`, `build_unroll_sample`, `collate_samples`. |
| `forexmind/muzero/replay.py` | `TrajectoryReplayBuffer`, `ReplayConfig`, sampling strategies, diagnostics, memory accounting. |
| `forexmind/muzero/collector.py` | `MuZeroCollector`, `CollectorConfig`, `CollectionStats`. |
| `tools/collect_muzero_trajectories.py` | Real-data collection smoke tool + report. |
| `tests/muzero_synthetic.py` | Shared synthetic-trajectory builder (not collected by pytest). |
| `tests/test_muzero_trajectory.py` | Trajectory contract and validation (20 tests). |
| `tests/test_muzero_targets.py` | Target construction and alignment (23 tests). |
| `tests/test_muzero_replay.py` | Replay sampling, capacity, leakage, diagnostics (24 tests). |
| `tests/test_muzero_collector.py` | Real-environment collection + batch inspection (14 tests). |
| `docs/stage43_muzero_replay.md` | This report. |

Changed: `forexmind/muzero/__init__.py` (exports), `README.md` (Stage 4.3
section). No existing Forex, training, or evaluation module was modified.

## 2. Trajectory data contract

```python
MuZeroTrajectory(
    observations,      # [T+1, obs_dim] float32
    actions,           # [T]        int64  (MuZero indices)
    rewards,           # [T]        float32
    root_policies,     # [T, 6]     float32
    root_values,       # [T]        float32
    action_masks,      # [T, 6]     bool
    terminated,        # [T]        bool
    truncated,         # [T]        bool
    planning_exposure, # [T+1]      float32
    planning_is_flat,  # [T+1]      bool
    boundary_value,    # float
    metadata,          # TrajectoryMetadata
    extra,             # dict
)
```

Storage is compact NumPy (float32 observations/targets, int64 actions, bool
masks), never nested Python objects per transition. **Observations are stored,
latent states are not**: latents go stale as weights change, so the learner must
recompute them with the current representation network.

`validate()` enforces every invariant below and raises a specific error
otherwise. It is called when a trajectory is built and again on `replay.add`.

## 3. Exact observation / action / reward indexing

For `T` real decisions:

```
observations    T + 1      observations[t] is the state BEFORE decision t
actions         T          actions[t] = a_t
rewards         T          rewards[t] = r_{t+1}   (the real environment reward)
root_policies   T          root_policies[t] = pi_t
root_values     T          root_values[t] = MCTS root value at s_t
action_masks    T          action_masks[t] = mask used at s_t
planning state  T + 1
```

The final observation exists because `a_{T-1}` produces `o_T`. `rewards[t]` is
the reward that `recurrent_inference(latent_t, a_t)` must predict — there is
**no** reward stored for `t = 0` itself, which is exactly why
`target_rewards` has length `K` while `target_values`/`target_policies` have
length `K+1`.

`terminated[t]`/`truncated[t]` describe the *outcome* of the transition taken at
state `s_t`, so they refer to `s_{t+1}`.

## 4. Replay architecture

```
TrajectoryReplayBuffer
├── add(trajectory, require_split="train")   # validates + FIFO evicts
├── sample(batch_size, target_config, strategy, rng, device) -> MuZeroBatch
├── decode_flat_positions(...)               # (traj_index, position) pairs
├── memory_report() / sampling_diagnostics()
└── SAMPLING_STRATEGIES = {"uniform", "decision_rich"}
```

A sampled training position is `(trajectory_id, position_t)`, from which the
learner obtains `observation_t`, `actions[t .. t+K-1]`, the policy targets for
`t .. t+K`, the value targets for `t .. t+K`, the reward targets for
`t .. t+K-1`, and all loss masks.

Sampling is position-level uniform over the flattened
`(trajectory, position)` space, decoded with a cumulative-offset array and
`np.searchsorted` (O(1) per sample, vectorised). Strategies are plain functions
registered in `SAMPLING_STRATEGIES`, so `prioritized` or `reanalysis` can be
added later without touching the buffer API.

**Leakage protection.** `add()` refuses any trajectory whose metadata split is
not the required one (default `train`), so validation/test data cannot enter
training replay. The collector independently asserts the sampler returned the
configured split.

## 5. Memory usage

Measured on real EURUSD/GBPUSD data (`obs_dim = 351`):

| Metric | Value |
|---|---:|
| bytes / transition | 1,545 |
| bytes / trajectory (T=16) | ≈ 24,700 |
| 4 trajectories, 64 transitions | 0.0989 MB |

Observations dominate (`(T+1) * 351 * 4` bytes), as expected. Growth is bounded
by `max_trajectories` (default 256) and optionally `max_transitions`, with FIFO
eviction; `memory_report()` reports counts, capacity usage, bytes per
trajectory/transition, and total MB. A test asserts that widening observations
visibly increases the estimate, so the accounting cannot silently drift.

## 6. Unroll length

`TargetConfig.num_unroll_steps = K`, default **5**. A sample at `t` provides
`actions[t .. t+K-1]` and targets for states `t .. t+K`, clamped at the
trajectory end and never crossing into another trajectory.

## 7. TD-step configuration

`TargetConfig.td_steps = n` (default 10) and `TargetConfig.discount`
(default 0.99). `use_boundary_value` (default `True`) toggles the documented
truncation bootstrap.

## 8. Value-target equation

```
z_t = Σ_{k=1..n} γ^(k-1) · r_{t+k}  +  γ^n · V_bootstrap(t+n)
```

Bootstrap resolution, in order:

| Situation | `V_bootstrap` |
|---|---|
| `t+n < T` (state inside the trajectory) | stored `root_values[t+n]` (the MCTS search value) |
| `t+n == T` and the episode ended | `boundary_value` |
| a true terminal was hit earlier | sum stops, bootstrap `0` |
| `use_boundary_value=False` | `0` at the boundary |

No arbitrary neural value is ever substituted: the boundary value is either
`0` (true terminal) or a genuine final MCTS root value recorded at collection
time for a truncated episode.

## 9. Policy-target construction

For unroll offset `k`, the policy target is the stored MCTS root visit
distribution `root_policies[t+k]`. If `t+k` is at or beyond the final recorded
observation — or the state is a true terminal — `policy_masks[t+k] = 0` and the
target is zeros rather than an invented distribution.

## 10. Reward-target alignment

```
initial_inference(o_t)   -> policy target pi_t,  value target z_t
recurrent(a_t)           -> reward target r_{t+1}, policy/pi_{t+1}, value/z_{t+1}
recurrent(a_{t+1})       -> reward target r_{t+2}, ...
```

`target_rewards[k]` corresponds to `actions[k] = a_{t+k}` and equals
`trajectory.rewards[t+k]` = the real `r_{t+k+1}`. The mandatory off-by-one
regression test uses `r = [11, 22, 33]` and asserts
`target_rewards == [11, 22, 33]` for the three recurrent steps.

## 11. Terminal / truncation handling

* **True terminal** (`terminated[-1]`): the reward sum stops at the terminal,
  the bootstrap is exactly `0`, and `validate()` rejects a non-zero
  `boundary_value`. The terminal state's policy target is masked out and its
  value target is `0`.
* **Truncation** (`truncated[-1]`): the episode was cut by the environment
  horizon and the task continues, so the bootstrap is the final MCTS root value
  recorded by the collector (`boundary_value`). Truncation is **never** treated
  as terminal.
* Bootstrapping is therefore allowed for every within-trajectory bootstrap state
  and for a truncated boundary; it is zero at a true terminal and zero outside
  the recorded data when `use_boundary_value=False`.
* `terminated` and `truncated` are stored separately and validated to never both
  be true on the same transition.

The one extra MCTS search needed for a truncated boundary is counted in
`CollectionStats.searches` (verified: `searches == env_steps + truncated
episodes`).

## 12. Action-mask handling

`action_masks` stores the real six-wide environment mask used at each decision
(after projection). `validate()` asserts, per step: correct dimension, HOLD
valid, at least one valid action, the selected action valid under its own mask,
and zero policy mass on invalid actions. Replay sampling re-checks the mask
integrity of every emitted batch.

## 13. Uniform replay-sampling behaviour

Position-level uniform is the canonical strategy. Every `(trajectory,
position)` pair with `0 <= position < T` is equally likely — HOLD-heavy states
are neither dropped nor down-weighted. A test samples 400 positions from a
12-transition trajectory and confirms every position is reachable, and another
confirms a fixed RNG seed reproduces the exact same `(trajectory_id, position)`
draws.

`decision_rich` exists but is inert unless `decision_rich_weight > 0`; with
weight `0.0` it is asserted to be byte-identical to uniform, so the baseline
stays measurable.

## 14. HOLD-frequency diagnostics

`replay.sampling_diagnostics()` reports:

```
actions: pct_hold, pct_flat, pct_short, pct_long   (sums to 1)
events:  pct_hold, pct_entry, pct_exit, pct_resize (sums to 1)
num_positions, sampling_strategy, decision_rich_weight
```

A deliberately HOLD-heavy trajectory (`LONG_50, HOLD, HOLD, HOLD, HOLD, FLAT`)
reports `pct_hold = 4/6`, keeps all four HOLD states sampleable, and is not
modified in any way.

## 15. Synthetic target tests

* Indexing: `o0 --a0/r1--> o1 --a1/r2--> o2 --a2/r3--> o3` stores
  `observations = 4`, `actions = 3`, `rewards = 3`, and the diagram's alignment
  holds exactly.
* n-step (§32): `γ = 0.9`, `r = [1, 2, 3]`, `V3 = 4`, `n = 3` →
  `z0 = 1 + 0.9·2 + 0.81·3 + 0.729·4 = 8.146` exactly.
* Terminal (§33): `o0 -> o1 -> terminal`, `n = 10` → `z0 = r1 + 0.9·r2`, no
  rewards beyond the terminal, bootstrap `0`, and `z_terminal = 0`.
* Truncation (§34): a truncated episode with `boundary_value = 5` gives
  `z0 = r1 + 0.5·r2 + 0.25·5`, and `state_is_terminal` is `False`.
* Policy alignment (§35): `argmax` of the targets at offsets 0/1/2 is 0/1/2 —
  no shift.
* Reward alignment (§36): `[11, 22, 33]` → `[11, 22, 33]`.
* Padding (§22): near the end, `reward_masks`, `policy_masks` and `value_masks`
  go to `0`, padded actions are `HOLD` with a legal HOLD-only mask, and the
  values are zeros — never another trajectory's data.
* Single-agent discounting (§23): `+γ` at every step; a mixed-sign example
  (`r = [1, -2]`) evaluates to `1 + 0.5·(-2) + 0.25·3`, with no sign flipping.
* Support compatibility (§24): targets are scalar economic values;
  `scalar_to_support` is a loss-layer concern and replay never stores support
  encodings.

## 16. Real-environment collection smoke test

`python -m tools.collect_muzero_trajectories --trajectories 4 --horizon 16
--simulations 8 --instruments EURUSD GBPUSD` (untrained model, CPU):

| Metric | Value |
|---|---:|
| trajectories | 4 |
| environment steps | 64 |
| mean length | 16.0 |
| terminated / truncated | 0 / 4 |
| MCTS calls | 68 (= 64 + 4 boundary searches) |
| recurrent inference calls | 544 (= 68 × 8) |
| mean reward | −0.00004964 |
| reward sum | −0.00317707 |
| replay memory | 0.0989 MB |
| bytes / transition | 1,545 |
| wall clock | 8.6 s |

Action frequencies: HOLD 25.00 %, FLAT 10.94 %, SHORT_100 15.62 %,
SHORT_50 18.75 %, LONG_50 14.06 %, LONG_100 15.62 %.
Decision mix: hold 25.0 %, entry 15.6 %, exit 10.9 %, resize 48.4 %.

Profitability is not judged and is meaningless here — the network is random.

Also asserted by the test suite: every collected trajectory validates, policies
sum to 1, HOLD is valid everywhere, planning metadata agrees with the live
account, and collection is bit-reproducible for fixed model/episode/search seeds.

## 17. Example sampled batch shapes

`K = 5`, `B = 8`, `obs_dim = 351`, 6 actions:

```
observation      [8, 351]
actions          [8, 5]        int64
target_rewards   [8, 5]        float32
target_values    [8, 6]        float32
target_policies  [8, 6, 6]     float32
policy_masks     [8, 6]        float32
value_masks      [8, 6]        float32
reward_masks     [8, 5]        float32
action_masks     [8, 6, 6]     bool
trajectory_ids   [8]           int64
positions        [8]           int64
```

All targets are finite, policy targets sum to 1 over valid actions, invalid
actions carry zero probability, and every row's real values are traced back to
its own trajectory and position in the tests.

## 18. Remaining concerns

- **Untrained model.** Search quality on the real network is meaningless, so the
  collected action mix reflects a random policy, not a strategy.
- **Value targets depend on search quality.** `root_values` come from MCTS over
  a random network; they are internally consistent but not informative yet.
- **No terminal modelling in search.** MCTS cannot see the episode horizon or
  liquidation boundary, so `root_values` near the end of an episode are
  optimistic. The boundary value only patches the final state.
- **Planning-chain drift.** The deterministic `after(action)` rule assumes the
  account reaches its target exposure exactly; under mark-to-market drift it can
  differ from reality. Stored planning metadata is *environment ground truth*,
  so trajectories stay correct, and the collector records
  `planning_chain_disagreements` in `extra` as a diagnostic.
- **Mark-to-market exposure drift** can also flip which exposure action is
  redundant near the masking tolerance; the live environment mask is always
  authoritative and stored.
- **Replay is in-memory only.** No disk serialization, no compression, no shared
  memory; a long run will hold everything in RAM.
- **Uniform sampling ignores learning progress.** No prioritization, no
  reanalysis, and no target recomputation as weights change.
- **`td_steps` and the boundary value are configured but untuned.**
- **Single-threaded collection.** One environment at a time; no actors.
- CUDA remains unavailable on this machine, so device parity is covered by
  skipped tests only.

## 19. Readiness for Stage 4.4 MuZero loss and learner

Yes. Everything the loss needs is available and verified:

- correctly indexed trajectories with explicit `T`/`T+1` contracts and a
  validating `validate()`;
- scalar `z_t` value targets, MCTS visit-distribution policy targets, and
  correctly aligned reward targets;
- explicit `policy_masks` / `value_masks` / `reward_masks` for padded and
  boundary positions, so the loss can be masked without special cases;
- `action_masks` per unrolled state for masked policy logits;
- batched `MuZeroBatch` tensors ready for the unroll, with `to(device)`;
- `trajectory_ids` and `positions` for later reanalysis or prioritization;
- replay bounded by capacity with `memory_report()`, and split-guarded so only
  train trajectories can enter;
- reproducible collection, so a training run can be replayed exactly.

Deliberately absent: optimizer, target network, loss, reanalysis, prioritized
replay, distributed actors, and any full training loop.
