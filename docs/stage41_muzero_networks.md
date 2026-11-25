# Stage 4.1 — MuZero core networks and inference API

Implementation date: 2026-09-10. This stage adds **only** the MuZero neural
architecture and its inference contracts. No MCTS, PUCT, root noise,
self-play, replay buffer, target generation, MuZero loss, reanalysis, or
training loop is implemented. The model has **not** learned a Forex world model;
the deliverable is a mathematically consistent, testable network interface that
Stage 4.2 MCTS can query.

Nothing in the frozen Forex contract changed: observation semantics, reward
(`reward_t = log(equity[t+1] / equity[t])`), execution timing, accounting,
currency conversion, episode mechanics, categorical actions, and validation
splits are untouched.

> **Stage 4.2 update.** MuZero was subsequently retargeted to its own frozen
> **six-action** space (`HOLD`, `FLAT`, `SHORT_100`, `SHORT_50`, `LONG_50`,
> `LONG_100`) while the shared ten-action environment stayed untouched. Every
> `num_actions`-dependent number below (policy width, parameter count, inference
> benchmark) has been refreshed for the six-action model; the network code
> itself is unchanged because `num_actions` was always a parameter. See
> [Stage 4.2 report](stage42_muzero_search.md) for the search layer.

---

## 1. Repository structure inspected

- `forexmind/observation/{schema,encoder}.py` — the observation contract.
  `ObservationSpec.encoded_shape` is the single source of truth for the flat
  vector size.
- `forexmind/environment/actions.py` — the categorical action system:
  `ACTION_NAMES` (10 entries), `TARGET_EXPOSURES`, `DISCRETE_ACTION_SIZE = 10`,
  and `valid_action_mask(exposure, is_flat)`.
- `forexmind/environment/forex_env.py` — `action_masks()` returns the
  state-dependent boolean mask; execution happens on the next M1 open.
- `forexmind/training/{config,networks,trainer}.py` — existing model-free
  Phase-3 conventions (`obs_dim = encoder.config.spec.encoded_shape[0]`,
  `MLP`, `count_parameters`, checkpoint/config serialization).
- `tests/test_training_ppo.py`, `tests/conftest.py`, `pyproject.toml` — test and
  tooling conventions (pytest, ruff, mypy; `train` extra carries torch).

Conclusion: the repository had no MuZero abstraction, so a new `forexmind/muzero/`
package was added rather than duplicating anything. It reuses the existing
action definitions and encoder spec directly instead of re-declaring them.

## 2. Files added / changed

Added:

| File | Purpose |
|---|---|
| `forexmind/muzero/__init__.py` | Public package surface. |
| `forexmind/muzero/config.py` | `MuZeroConfig` + `observation_dim()` derivation. |
| `forexmind/muzero/types.py` | `NetworkOutput` shared result type with shape validation. |
| `forexmind/muzero/support.py` | `scalar_to_support` / `support_to_scalar`. |
| `forexmind/muzero/networks.py` | `RepresentationNetwork`, `DynamicsNetwork`, `PredictionNetwork`, `parameter_report`. |
| `forexmind/muzero/inference.py` | `MuZeroNetwork` API, `apply_action_mask`, `masked_policy_probs`. |
| `tools/benchmark_muzero_inference.py` | Parameter report + inference throughput benchmark. |
| `tests/test_muzero_support.py` | Support-transform tests (16). |
| `tests/test_muzero_networks.py` | Architecture/inference tests (59). |
| `docs/stage41_muzero_networks.md` | This report. |

Changed: `README.md` (short Stage 4.1 section + benchmark command). No existing
Forex, training, or evaluation module was modified.

## 3. MuZero architecture implemented

```
observation
    ↓  h_theta  RepresentationNetwork : Linear→LayerNorm→SiLU (x2) → Linear → LayerNorm
latent state s_t^0  [B, latent_dim]
    ↓  f_theta  PredictionNetwork : policy trunk → 6 logits ; value trunk → value head
policy logits [B, 6] , value [B, 1]

latent state s + action index
    ↓  nn.Embedding(num_actions, action_embedding_dim)
    ↓  DynamicsNetwork : MLP → transition head (delta) + reward head
next latent [B, latent_dim] = LayerNorm(s + delta)   (residual, configurable)
reward head output → reward [B, 1]
```

- `RepresentationNetwork` is a compact MLP; no transformer/recurrent/attention
  stack, because the current Phase-2 observation is a flat vector, not a
  sequence.
- `DynamicsNetwork` predicts its own latent transition and reward, never raw
  market observations, and never touches the environment.
- `PredictionNetwork` returns **raw** logits; softmax is applied only outside
  the network (search/loss).
- `residual_dynamics`, `layer_norm`, and the activation are configurable.

### Leakage audit (no hidden information)

`initial_inference` accepts exactly one tensor — the flat encoded observation
that the Phase-2 encoder already produces causally (`market` window returns +
account state + time features + instrument one-hot). It has no access to
`info`, the environment object, or the dataset. `recurrent_inference` accepts
only a latent state and an action index; it never calls `env.step()` and cannot
read future candles, future execution prices, future rewards, future equity,
next observations, or validation labels. Masking uses only the causal
`action_masks()` snapshot. Therefore the representation path cannot leak
information unavailable at decision time.

## 4. Observation dimension actually used

`obs_dim = 351`, **derived** at runtime from the encoder’s `ObservationSpec`,
not hard-coded:

```
64 context bars * 5 market features = 320
+ 10 account features
+ 14 time features
+  7 instrument one-hot
= 351
```

`MuZeroConfig.from_encoder_config()` / `observation_dim()` call
`EncoderConfig().spec.encoded_shape[0]`, so a context-length change propagates
automatically (verified by `test_config_from_encoder_config_sets_obs_dim`).

## 5. Latent dimension

`latent_dim = 128` (configurable; 256 is a supported one-line change). The
latent state is `LayerNorm`-normalized inside the representation network and
again after each residual dynamics step. Normalization is per-sample (over the
latent features), never across unrelated batch samples.

## 6. Hidden / action embedding dimensions

| Setting | Value |
|---|---|
| `num_actions` | 6 (MuZero space, see Stage 4.2) |
| `hidden_dim` | 256 |
| `num_layers` | 2 hidden blocks per sub-network |
| `action_embedding_dim` | 16 |
| `activation` | SiLU |
| `value_support_size` | 21 |
| `reward_support_size` | 21 |

## 7. Scalar vs support-based reward/value design

Support-based (MuZero-style) categorical prediction is the default
(`use_support=True`), using the preferred option from the brief:

- `scalar_to_support(x, support_size)` produces a two-point probability
  distribution over 21 bins spanning `[-1, 1]`.
- `support_to_scalar(logits, support_size)` decodes the expected support value
  with the invertible epsilon expansion.
- The prediction value head outputs 21 logits; the dynamics reward head outputs
  21 logits. `NetworkOutput` therefore carries `value`/`reward` **scalars**
  `[B, 1]` *and* the raw `value_logits`/`reward_logits` `[B, 21]`.
- `use_support=False` degrades both heads to scalar regression (output width 1),
  and the public inference API is byte-for-byte identical
  (`value_logits=None`, `reward_logits=None`). Scalar and support predictions are
  never mixed within one model — the mode is fixed by config and reported in
  the benchmark JSON.

Reward targets keep the unchanged Forex scale; only the *representation* of the
target is configurable. `value_scale` / `reward_scale` exist so a natural-scale
factor can be introduced later without changing the API.

For initial inference `reward = 0` exactly and `reward_logits = None`, because
no dynamics transition has occurred.

## 8. Parameter count (defaults)

| Component | Parameters |
|---|---:|
| Representation | 190,080 |
| Dynamics | 142,581 |
| Prediction | 206,619 |
| **Total** | **539,280** |

(Six-action model. The ten-action model measured 540,372.)

## 9. Initial inference example

`initial_inference(observation, action_mask=None)` with `B = 4`:

```
observation     (4, 351)
latent_state    (4, 128)
policy_logits   (4, 6)
value           (4, 1)
reward          (4, 1)      # exactly 0
value_logits    (4, 21)
reward_logits   None
```

`[B, 351] → [B, 128] → [B, 6] / [B, 1]`. A single observation of shape
`(351,)` is accepted and treated as `B = 1`; NumPy `float32` arrays are accepted.

## 10. Recurrent inference example

`recurrent_inference(latent_state, action, action_mask=None)` with `B = 4` and
actions `[0, 3, 4, 5]`:

```
latent_state    (4, 128)    # the NEXT latent state
policy_logits   (4, 6)
value           (4, 1)
reward          (4, 1)
value_logits    (4, 21)
reward_logits   (4, 21)
```

`action` is the discrete index `0…5` (not target exposure); a Python int is
broadcast across the batch, a tensor must be `[B]` (or `[1]`). The returned
`latent_state` is the successor so MCTS can chain calls. This path is a pure
neural transition and never calls `env.step()`.

## 11. Action-mask behavior

Masks are applied in the inference layer only (`apply_action_mask`), so the
prediction network keeps returning raw logits and training can still read them
by passing `action_mask=None`.

- Invalid actions are filled with `torch.finfo(dtype).min`, so their softmax
  probability is exactly `0.0`.
- `HOLD` (index 0) must remain valid, and every batch row must keep at least one
  valid action — both enforced with clear `ValueError`s.
- Requirement `mask.shape == [B, num_actions]`; a 1-D mask is broadcast.
- The mask is not applied in place (`test_apply_action_mask_is_pure_function`).

Using the frozen environment mask (`valid_action_mask`), projected onto MuZero's
six actions:

```
flat account        -> FLAT masked (other 5 valid)
exposure = +0.50    -> LONG_50 masked
masked probability of LONG_50 = 0.0 ; HOLD prior = 0.1965 (32-sim search)
```

## 12. Unit-test results

New tests: **70 passed, 1 skipped** (CUDA-only test skipped on this CPU box).

| Area | Coverage |
|---|---|
| Representation | shape/finite for B=1,4,32; wrong-obs-dim rejection |
| Initial inference | `reward == 0`; `[B,6]` policy; finite value/latent; 1-D and NumPy inputs |
| Recurrent inference | all 6 actions finite; per-batch actions; pure-neural call on a synthetic latent |
| Action encoding | 6 distinct embeddings; out-of-range indices raise; element-wise range check; float rejection |
| Masking | zero invalid probability; HOLD always valid; ≥1 valid action; shape/broadcast; purity |
| Gradients | synthetic policy+value+reward loss → every parameter finite grad, no NaN/Inf; reward-head gradient |
| Batch/device | B=1,4,32; outputs stay on the model device; CUDA test gated on availability |
| Unroll | 5-step stability, divergence between action sequences, order sensitivity |
| Reporting | parameter report sums; scalar-head variant shares the API |
| `NetworkOutput` | shape validation for `value`, `policy_logits`, `value_logits` |
| Support | distribution validity, endpoints, centre bin, clamping, round-trips, validation |

Full repository suite after the change: **511 passed, 2 skipped** (Stage 4.1);
**586 passed, 2 skipped** after Stage 4.2.
`ruff check`, `ruff format --check`, and `mypy` (105 files) are clean.

## 13. Five-step latent unroll result

```
o_t → initial_inference → s0
    → a0=0 → recurrent_inference → s1
    → a1=3 → recurrent_inference → s2
    → a2=4 → recurrent_inference → s3
    → a3=5 → recurrent_inference → s4
    → a4=2 → recurrent_inference → s5
```

With `B = 2`: six latent states (one initial + five recurrent), each `(2, 128)`;
latent, policy, value, and reward stay finite at every step; no NaN/Inf appears.
For the same root latent, `[2,2,2,2,2]` and `[5,5,5,5,5]` converge to different
final latents, and `[1,5]` differs from `[5,1]`, so actions genuinely affect the
transition.

## 14. Inference throughput benchmark

`python -m tools.benchmark_muzero_inference --batch-sizes 1 16 64 256 --iters 300 --warmup 30`
(CPU-only AMD64, `torch 2.13.0+cpu`, latent 128, hidden 256×2, support 21):

| Batch | initial calls/s | initial states/s | recurrent calls/s | recurrent states/s |
|---:|---:|---:|---:|---:|
| 1 | 343.7 | 343.7 | 247.7 | 247.7 |
| 16 | 256.3 | 4,101.0 | 226.9 | 3,630.8 |
| 64 | 203.6 | 13,027.9 | 171.7 | 10,987.3 |
| 256 | 119.5 | 30,603.7 | 107.7 | 27,581.6 |

`recurrent_inference` is the MCTS-critical call; at batch 64 it sustains
~11.0k states/s (~15.5k latent-expansions/min) on this machine single-threaded.
Run-to-run spread on this shared box is roughly 1.6×, so treat these as a
baseline order of magnitude, not a precise rating. Results are written to
`data/reports/stage41_muzero_inference_benchmark.json`. No optimization was
attempted.

## 15. Remaining issues

- No training exists, so value/reward scales are still nominal; the support
  ranges (`[-1, 1]` scaled) will need to be re-derived from observed target
  statistics once real self-play data exists.
- The reward head predicts a scalar per transition; multi-step reward
  accumulation, discounting, and target construction belong to the later loss
  stage.
- Latent normalization was chosen for stability but there is no empirical
  evidence yet about representation collapse or gradient conditioning.
- `recurrent_inference` re-embeds the action each call; a small caching layer
  may be worth adding only if profiling shows it matters.
- CUDA was unavailable on this machine, so the device parity path is covered by
  a skipped test only.
- The observation input dimension is derived correctly, but the MuZero
  representation still consumes the flat `(351,)` vector; a structured temporal
  encoder is a later, separate decision.

## 16. Readiness for Stage 4.2 MCTS

Yes. The API MCTS needs already exists and is stable:

```python
root  = model.initial_inference(obs, mask)
child = model.recurrent_inference(root.latent_state, action, next_mask)
```

- One shared `NetworkOutput` type, one batched code path (B = 1 and B > 1),
  action indices aligned with MuZero's frozen six-action space (Stage 4.2).
- `recurrent_inference` is a pure neural latent transition — it never calls the
  environment, so latent planning cannot leak real transitions.
- Raw logits are returned and masking lives in the inference layer, which is
  exactly where PUCT priors will be computed.
- Determinism is verified for fixed parameters/inputs, and no sampling happens
  inside the network, so search owns all stochasticity.

MCTS is intentionally **not** implemented here.
