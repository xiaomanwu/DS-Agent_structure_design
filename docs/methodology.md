# Methodology

What follows is what the code actually implements, including the choices that are easy to get wrong.
Every section names the module that owns it, so a reviewer can jump straight to the source.

---

## 1. Metric anomaly detection (`ds_agent/tools/anomaly.py`)

### 1.1 Baseline

For a daily metric, day *t* is compared against

```
expected_t = level_t × seasonal_{dow(t)}
```

- **`seasonal`** is a stable day-of-week index: each day divided by the centred 7-day median, then the
  median per weekday, normalised so the week averages 1, refined twice by the median residual per
  weekday. A fixed profile is deliberate: STL's locally-varying seasonal component drifts, and a
  drifting Saturday index generates phantom anomalies.
- **`level_t`** is the median of the deseasonalised values over the *previous* 28 days, shifted by one
  day so today can never enter its own baseline.

Two subtleties matter:

1. **A trailing window describes the past, not today.** The median of days `t-28..t-1` is centred
   around `t-14`, so in a business growing 0.15%/day the baseline is ~2% low — several sigma on a
   low-noise metric, and every day looks slightly high. The code adds the local growth rate back:

   ```
   level_t = median(deseason_{t-28..t-1}) × exp(slope × 14)
   slope    = median of 3 overlapping 56-day log differences  (robust to one bad day)
   ```

2. **A centred smoother hides the incident it should find.** Both STL and a centred rolling median see
   the incident days and pull the baseline down with them, so a 10% drop ends up as a 3% residual.
   This is why `baseline="stl"` exists but is not the default.

### 1.2 Residual scale

```
residual_rel = actual / expected - 1
sigma = max(1.4826 × MAD, 0.8 × std(inliers))     inliers = within 5 × 1.4826 MAD
z = residual_rel / sigma
```

Pure MAD is the right default, but a robust fit can shrink it; the std floor prevents false alarms.
Computing that floor on **inliers** prevents a segment that is down 40% for a week from inflating its
own noise scale and hiding itself — the exact failure mode that makes naive z-scores useless on
segment scans.

### 1.3 Point anomalies and level shifts

- A **point anomaly** is `|z| ≥ 3.5` and `|residual| ≥ 5%`.
- A **level shift** is found by two-sided CUSUM on `z` (slack `k = 0.5`, threshold `h = 5`). A 10% drop
  sustained for a week is only ~2.5σ on any single day — statistically invisible point-wise, yet it is
  the most common shape of a real business incident. CUSUM accumulates the evidence, so it fires after
  2–3 days.
- Alarm runs are extended **backwards to the onset** (while the residual still points the same way and
  stays above 35% of the run mean) and validated with a one-sample t-test on the run (`p < 0.05`, at
  least 2 days, at least 5% impact). Runs failing the gate are discarded, which is what keeps a noisy
  flat series quiet.
- Contiguous flagged days merge into **episodes**, each reporting actual vs expected, impact, mean
  deviation, p-value, severity and `kind` (`point` vs `level_shift`). Severity comes from the segment
  t-statistic, so a consistent 5% bleed for 11 days outranks a single noisy spike.

### 1.4 Comparing two windows

`compare_windows` defaults to a **paired** t-test when both windows have the same length. Windows that
are an integer number of weeks apart line up by weekday, so pairing removes the weekly component
entirely. The unpaired test pays for that variance: in a 7 vs 7 comparison the weekly swing alone is
~8%, which buries a real 7% drop (p ≈ 0.10 instead of p ≈ 0.002).

For the same reason, the diagnose handler judges significance against the **seasonal baseline**
(`actual vs expected`, paired by day) rather than against the previous week: when an incident is
already 10 days old, the prior week is contaminated and a week-over-week test understates the problem.

### 1.5 Time-series baselines (`ds_agent/tools/forecast.py`)

The decomposition above is fast and explainable, but it is not the only way to answer "what should
today have been". Seven models are available as the baseline, and the question can name one
("用 holt-winters 看下异常") or ask for the comparison ("时序基线对比"):

| `baseline` | Model | When it earns its keep |
| --- | --- | --- |
| `trailing` (default) | weekday profile × causal trailing level | fast, explainable, robust to a single bad day |
| `holt_winters` | damped additive Holt-Winters, `m = 7` | trend + stable weekly shape, level shifts need to be visible |
| `holt_winters_mul` | multiplicative seasonal component | weekly amplitude grows with the level (most GMV series) |
| `arima` | SARIMAX(1,1,1)(0,1,1)₇ | autocorrelated residuals; reporting the model, not the decomposition |
| `stl_ets` | robust STL seasonal indices + ETS on the adjusted series | noisy seasonal shape, wants ETS damping |
| `auto` | backtest all candidates on one rolling origin | the defensible default when the user asks "which baseline" |

Three properties make these usable rather than decorative:

1. **They stay causal.** Every forecast for day *t* is fitted on data `< t` (`rolling_forecast` with a
   refit cadence, default weekly). A model refitted on the full series would smooth the incident into
   its own expectation and report a smaller anomaly the longer it lasts.
2. **Selection is measured, not asserted.** `select_method` computes MASE on the same rolling origin for
   every candidate and returns the winner plus the full table and the margin over the runner-up. MASE
   (MAE over the in-sample seasonal-naive error) is used because "better than repeating last week" is
   the comparison that means something to a business reader; a 1% MASE gap does not justify three extra
   assumptions, and the margin is reported so that judgement is possible.
3. **The chosen model reports its own error.** With an explicit `baseline=`, the one-step-ahead MAE /
   RMSE / MASE of *that* model is computed and shown next to the finding. A baseline with a bad MASE
   produces confident nonsense, and that belongs on the same screen as the anomaly.

The agent surfaces all of this: a KPI for the chosen baseline, the backtest table when `auto` was
requested, and a sentence naming the model and its MASE. `trailing` remains the default because a
fitted model can absorb the incident through its level/trend state — the very failure mode a smoother
has — and because batch segment scans call it thousands of times.

---

## 2. Contribution and attribution (`ds_agent/tools/contribution.py`)

### 2.1 Additive metrics

```
Δtotal = Σ_i (v_i,current − v_i,baseline)
contribution_i = Δv_i
share_i = Δv_i / Δtotal
```

Exact, with no residual. New and lost segments need no special case: a segment that disappeared simply
contributes its full baseline value.

### 2.2 Ratio metrics

With `R = Σ w_i r_i` (`w_i` = volume share, `r_i` = segment rate):

```
ΔR = Σ_i w_i,base · Δr_i        rate effect   (behaviour within segments)
   + Σ_i r_i,base · Δw_i        mix effect    (traffic composition)
   + Σ_i Δw_i · Δr_i            interaction
```

The three terms sum exactly to `ΔR`; the code asserts the residual is ~0. This is what lets the
narrative say "conversion fell 4%: −3.1pp from iOS converting worse, −0.7pp from traffic shifting
toward Android".

### 2.3 Multi-dimension attribution

"Peeling": every dimension is decomposed, then ranked by

```
score = |largest segment contribution| × (0.5 + 0.5 × top1_share)
```

Absolute size matters, but a dimension where one segment carries the move is a better explanation than
one where it is spread thin. The top dimension's leading segment is then decomposed along the
remaining dimensions. Each step reports the share of the total move it explains, so a partial answer is
visibly partial.

`focus` pins the primary dimension when the user asked for one ("break it down by category"). Answering
with `device` because it scores higher is technically better and practically useless.

---

## 3. Experiment design and readout (`ds_agent/tools/experiment.py`, `ds_agent/stats/`)

### 3.1 Power and sample size

One solver (`_solve`, Brent) inverts the same `_power_two_sided` function for every metric family, so
`power()` and `sample_size()` cannot disagree:

```
power = Φ(|δ|/SE − z_{1−α/2}) + Φ(−|δ|/SE − z_{1−α/2})
```

- proportion: `SE = √(p₁(1−p₁)/n₁ + p₂(1−p₂)/n₂)`
- mean: `SE = sd·√(1/n₁ + 1/n₂)`
- count: Poisson variance, with an over-dispersion multiplier
- ratio metric: delta method on `Σnum/Σden`

Duration rounds up to whole weeks (never fewer than 7 days), and cluster designs inflate the sample by
`1 + (m − 1)·ICC`.

### 3.2 Readout

1. **SRM first.** Chi-square on the arm split, flagged at `p < 0.001` — not 0.05: SRM is a pipeline bug,
   and a 0.05 threshold would cry wolf every twentieth experiment.
2. **Per-metric tests.** Two-proportion z (pooled SE for the test, unpooled for the CI, plus a
   log-scale relative-lift CI); Welch t with a bootstrap percentile CI for heavy-tailed revenue; delta
   method for ratio metrics, including the numerator/denominator covariance.
3. **CUPED.** `θ = Cov(y, x)/Var(x)`, `y_adj = y − θ(x − x̄)`; the achieved variance reduction is
   reported, not assumed.
4. **Multiplicity.** Benjamini–Hochberg across the metric family; the ship decision uses `p_adjusted`.
5. **Heterogeneity.** Per-segment effects, explicitly labelled exploratory (no correction).
6. **Sequential looks.** O'Brien–Fleming alpha spending for planned interim analyses, and an mSPRT
   always-valid p-value for continuous monitoring.

### 3.3 The 12 archetypes

`ab`, `abn` (A/B/n), `factorial`, `stratified`, `cluster`, `switchback`, `holdout`, `bandit`,
`sequential`, `cuped`, `triggered`, `quasi`, `nested` (parent-child), `reversal` (swap-back),
`two_sided` (2×2 buyer × seller). Each carries when to use it, the randomisation unit, the analysis
plan and its characteristic pitfalls — in both Chinese and English, because the recommendation is read
by humans, not parsed.

Type selection is keyword-driven with marketplace-specific weighting: two-sided marketplaces get
switchback (time-based randomisation) rather than cluster, since the treatment spills through supply.

### 3.4 Designs that are not a plain A/B (`ds_agent/tools/experiment_designs.py`)

Each of these exists because a standard user-level split cannot answer the question. None of them is a
variation on the same estimator; the inference unit changes, which is the whole point.

#### Switchback (time-sliced rotation)

Markets take turns: all markets get the treatment for a block of days, then all get control, repeated.
The **inference unit is the time block**, not the market-day. Treating market-days as independent
multiplies the nominal sample size by the number of markets and produces absurd p-values, because
every market shares the same daily shock.

```
y_t = α + τ·arm_t + trend_t + DOW_t + ε_t,    SE: Newey-West (HAC), lag ≈ 1
```

- **Washout**: the switch day itself mixes both arms (a user treated at 09:00 and controlled at 10:00).
  One day is dropped per switch by default; the reply reports `washout_days` so the choice is visible.
- **Paired blocks**: adjacent block pairs are differenced, which removes any slow drift shared by the
  two blocks. It is reported next to the HAC estimate; when the two disagree, the drift is not slow.
- **Carryover test**: yesterday's arm is added as a regressor. A coefficient that is a large fraction
  of τ means the washout is too short, and the handler says so rather than reporting the inflated
  effect.

#### Reversal (swap-back)

Two phases with the arms swapped at a midpoint. It tests the *pipeline*, not just the effect:

- `τ₁` is the effect before the swap, `τ₂` after. A real effect flips sign.
- **Sum test**: `τ₁ + τ₂ ≈ 0` is the drift check. If the two phases do not sum to zero the metric or the
  instrumentation moved, and the original estimate cannot be trusted (`infrastructure_valid`).
- **Persistence**: `|τ₂| / |τ₁|`. Near 1 means the effect is a stable property of the treatment; well
  below 1 means it decays (novelty effect, or a novelty-hungry segment).
- A pooled symmetric estimate is reported for when symmetry holds.

#### Parent-child (nested / orthogonal layers)

A parent experiment (page layout) and a child experiment (button copy) run on the same users. Two
questions: does the child still read correctly, and does the pairing interact?

- **Orthogonality**: chi-square on the parent × child cross-tab, flagged below `p < 0.01`. With
  hash-based assignment this should pass by construction; when it fails, the bucketing is wrong.
- **Parent fixed-effect adjustment**: `y = τ·child + α_parent + ε` with HC1 errors. Without it the
  child estimate absorbs the parent's effect through unbalanced cells.
- **Interaction**: a `parent × child` term. Non-significance is not proof of no interaction, so the
  estimate and its CI are reported alongside the p-value.
- A cell under 1,000 users refuses to run: with four cells and a skewed metric, the analysis is noise.

#### Two-sided (2×2 buyer × seller)

Four cells (`00`, `10`, `01`, `11`): neither side treated, buyers only, sellers only, both. One
difference-in-differences cannot separate a direct effect from cross-side spillover, so the design is
read from the marginals:

- **Direct effects**: `buyer_direct = mean(10) − mean(00)` on a buyer-side metric,
  `seller_direct = mean(01) − mean(00)` on a seller-side metric. Cross-side spillover is the *other*
  marginal (`mean(01) − mean(00)` on a buyer metric) and is reported as a ratio to the direct effect.
- **Interaction**: `(11 − 10) − (01 − 00)`. A negative interaction means treating both sides is worse
  than the sum of the parts — usually a pricing or inventory squeeze.
- **Market-level bootstrap** for the CI, because buyers inside a market are not independent draws.

#### Sensitivity analysis

The question "is this robust?" is answered by re-estimating under every defensible alternative and
showing the spread, not by repeating the same test:

winsorise 1%/5%, trim 1%/5%, drop the top 0.1%, `log1p`, CUPED, segment fixed effects, bootstrap CI —
each only where it is meaningful (a 0/1 conversion metric has nothing to winsorise or log, so those
specifications are skipped rather than faked). Two extra checks:

- **A/A placebo**: the same pipeline run on a random split of the control arm. The false-positive rate
  across replications should be near α; a much higher rate means the test itself is miscalibrated.
- **Segment jackknife**: drop one segment at a time and re-estimate. One segment carrying the whole
  effect is a finding about that segment, not about the treatment.

The verdict is stated in terms of sign stability and the width of `lift_range`, because "significant in
3 of 3 specifications" and "significant in 3 of 9" read identically if only the count is reported.

---

## 4. Causal inference (`ds_agent/tools/causal.py`, plus `uplift.py`, `rdd.py`, `ts_causal.py`)

### 4.1 Difference-in-differences

```
y_it = β·(treated_i × post_t) + α_i + γ_t + ε_it
```

Unit and time fixed effects absorb the main effects, so only the interaction is estimated. Standard
errors are clustered at the unit level by default.

The relative effect is normalised on the **treated counterfactual**, not the control mean:

```
relative = y₁₀ × (y₀₁/y₀₀) / y₁₀ − 1
```

With `log_outcome=True` the model is estimated in logs and `exp(β) − 1` is reported directly. The two
differ by the base effect, and the report says which one was used rather than silently picking one.

### 4.2 Event study

Pre-period leads test parallel trends: a joint F-test on all leads, plus per-period coefficients and a
plot. If the joint test rejects, the DiD estimate is reported with an explicit warning instead of being
presented as causal.

### 4.3 Matching and weighting

- **PSM**: logistic propensity model, 1:1 nearest neighbour on the logit with a caliper of
  `0.2 × sd(logit)`, sort + binary search so it scales to 10⁵ rows, bootstrap CI (200 reps) and
  standardised mean differences for balance.
  - *Principle.* The propensity score `e(x) = P(T=1|x)` collapses all covariates into one balancing
    number: conditional on `e(x)`, treated and control units are exchangeable (Rosenbaum & Rubin).
    Matching on the score therefore removes the *observed* selection bias; the estimate is the **ATT**
    — the effect on the treated, not the ATE.
  - *Use when.* No experiment exists, treatment was assigned by people (manual targeting, self-selection),
    you have user-level covariates, and both arms are reasonably large.
  - *Diagnostics shipped.* The tool reports the **common support** interval (overlap of the two
    propensity distributions) and how many treated units fall outside it — their effect is simply not
    identified by matching. An overlap histogram of the two arms is rendered next to the estimate.
    Balance is judged by |SMD| < 0.1 after matching, per covariate.
  - *Does not fix.* Unobserved confounding ("the boss picked them"), and it never turns a
    non-random assignment into a random one — it only balances what you measured.
- **IPW**: Hájek (normalised) weights with trimming of the top/bottom 1%.
- **AIPW**: doubly robust — a linear outcome model plus inverse-propensity weights, so the estimate is
  consistent if either the outcome model or the propensity model is right.

All three report the covariate set and the overlap diagnostics, because "the estimate moved" is only
interpretable alongside the assumptions. Run PSM and AIPW together: agreement is evidence, divergence
usually means poor overlap.

### 4.4 Synthetic control

Weights `w` minimise standardised pre-period RMSE subject to `w ≥ 0`, `Σw = 1` (SLSQP). Inference is
**in-space placebo**: the same fit is run on every donor, the treated unit's post/pre gap ratio is
ranked against theirs, and `p = rank/(n+1)` — a floor of `1/(n+1)`, because with 10 donors you cannot
claim p < 0.1. Donors must exclude other treated units, otherwise the donor pool absorbs the treatment.

`compare_estimators` runs several of these on the same data and reports the spread; disagreement is
information, not noise to be averaged away.

### 4.5 Uplift: who responds, not whether it worked (`ds_agent/tools/uplift.py`)

An ATE is the wrong number for targeting. The question is "which users move because of the treatment",
and that needs a model of the *effect*, not of the outcome.

- **Causal tree.** Splits maximise the difference in treatment effect between children, with a
  minimum leaf size and a depth cap:
  `Δ = |τ(left) − τ(right)|`, `τ(node) = mean(y|T) − mean(y|C)`. Splitting on outcome fit (a normal
  regression tree) finds the users who convert, not the users who convert *because* of the promo.
- **Causal forest.** Many honest trees: each tree is fitted on a subsample and its leaves are evaluated
  on held-out data, so the CATE of a leaf is not the CATE it was chosen for. `cate` is the average over
  trees.
- **Qini curve.** Sort users by predicted CATE, accumulate incremental conversions against the
  random-targeting diagonal. `Qini = 0.34` on the demo data means 34% of the treatment's total effect is
  captured by ordering users the model's way instead of at random. Deciles and a targeting table
  (top 10/20/30/50% capture share) translate that into a decision.
- **Cross-fitted AIPW.** The ATE is re-estimated with doubly robust scores to check the forest's
  average against a semiparametric estimator; a large gap means the forest is fitting noise.
- **T/S/X learners** are reported for comparison. They disagree in useful ways: a two-model learner
  amplifies noise in the outcome model, which is exactly what the honest split in the forest removes.

### 4.6 Regression discontinuity (`ds_agent/tools/rdd.py`)

Treatment switches at a threshold of a continuous score, so units just above and just below are
comparable. Three things decide whether the estimate means anything, and all three are reported:

1. **Local linear, not local constant.** A kernel mean is biased whenever the outcome has a slope in
   the running variable; the linear term removes that first-order bias. Triangular kernel, HC1 errors
   (heteroskedasticity is the norm around a cutoff), estimate evaluated at the cutoff from each side.
2. **Bandwidth.** CCT-style rule of thumb `h = 5.4·sd(x)·n^(−1/5)`, refined by leave-one-out CV over a
   grid. The estimate is then re-run at 0.5h…1.5h; `bandwidth_stable` is false when the sign flips,
   which is a statement that the design cannot answer the question.
3. **No manipulation.** A Poisson density test on counts either side of the cutoff. If units can sort
   across the threshold (a score you can game), the comparison is between two self-selected groups and
   nothing downstream can rescue it.

Plus the cheap falsification tests: **covariates that cannot respond to treatment must not jump**
(a significant jump falsifies the design), and **placebo cutoffs** at ±0.5h, where a "jump" would mean
the functional form, not the treatment, is producing the discontinuity. With `fuzzy=`, the estimate
becomes the Wald ratio `jump(outcome) / jump(treatment)` — a LATE for compliers at the cutoff, with the
first-stage jump and compliance rate reported.

### 4.7 Time-series causal: one unit, before and after (`ds_agent/tools/ts_causal.py`)

When there is no control group — a launch, a policy, a site redesign — the counterfactual has to be
forecast from the pre-period. Two estimators, deliberately different:

**(a) ARIMA interrupted time series.**

```
f(y_t) = β·post_t + γ₁·trend_t + γ₂·DOW_t + γ₃·controls_t + u_t,    u_t ~ ARMA(p,q)
```

- **Log scale** by default: a multiplicative weekly pattern and a proportional shock become additive.
- **The trend and day-of-week terms are not optional.** Without the trend, β absorbs the growth that was
  happening anyway — on the demo series that turns a true +12% into +24%. Without the weekday terms the
  post-period mix of weekdays biases β.
- **The intercept is not optional either.** With `d = 0` and no constant, the series level has nowhere to
  go and loads onto the trend/post columns; the same specification then produced +12% and −13.5% on
  consecutive runs. The design always carries `_const` and `SARIMAX(trend="n")`, and the start parameters
  come from OLS (statsmodels' zero start occasionally settles in a wrong mode).
- **Inference uses the estimated error structure.** OLS with a post dummy on a weekly series inflates
  the t-statistic by an order of magnitude. Ljung-Box on the residuals is reported: `p < 0.05` means the
  SE is still too small and the report says so instead of quoting the p-value.
- **Placebo interventions** are fitted at 8 fake dates inside the pre-period. If the real effect is not
  larger than those, it is not distinguishable from the series' own wiggles.
- A **specification curve** (with/without trend, with/without weekday terms, raw-scale seasonal ARIMA,
  level vs log) shows whether the answer is an artefact of one design choice.

**(b) Structural counterfactual ("causal impact").**

A local level state-space model is fitted on the pre-period with standardised control series and
weekday dummies, then forecast over the post-period; the point forecast is the counterfactual and the
state-space variance gives an interval that **widens with the horizon** instead of pretending the
counterfactual is known.

- **Stochastic trend off by default.** Fitted on a trending series it extrapolates the trend and eats
  the level shift (it halves the +12% effect on the demo data). It is exposed, not hidden.
- **Controls are standardised on the pre-period only** — standardising on the full series leaks
  post-period information into the counterfactual.
- **Pre-period fit is reported as one-step-ahead R²/RMSE**, with the diffuse-initialisation first point
  dropped (keeping it yields R² = −5.9 on a model that is actually good).
- Placebo windows inside the pre-period ask a different question from the model's own interval: how
  often does a window this long look like this "effect"?

`compare_ts_methods` puts the naive pre/post difference next to both models. The naive number is what a
dashboard shows, it bills the trend to the intervention, and the gap between it and the models is
usually the most persuasive number in the room.

---

## 5. Choosing a method (`ds_agent/tools/method_advisor.py`)

The advisor parses a scenario into features (experiment or not, treatment mechanism, data shape,
goal), then applies a decision tree over the whole method catalogue and returns ranked
recommendations with reasons, assumptions, pitfalls — and the name of the tool that implements each.

| Scenario | First choice | Why | Runner-up for cross-checks |
| --- | --- | --- | --- |
| Randomisable units, one change | Standard A/B (+ CUPED) | Randomisation is the only assumption-free identification | A/A + spec curve |
| Spillover across users (marketplace) | Switchback or two-sided 2×2 | User-level splits are contaminated by cross-side effects | Cluster bootstrap |
| Two experiments stacked on one traffic pool | Parent-child nested | Orthogonality must be verified, parent FE absorbed | Chi-square + interaction |
| Treatment = threshold on a score | RDD | Near the cutoff the design is a local random experiment | Density + bandwidth sensitivity |
| Policy rolled out across units over time | DiD + event study | Panel with clean controls; pre-trends testable | Synthetic control (single treated unit) |
| One unit, known intervention date, no controls | ARIMA ITS / structural counterfactual | Trend and seasonality must be modelled or they are billed as effect | Placebo dates, control series |
| Treatment assigned by people, cross-section | PSM → IPW/AIPW | Balance what was observed; doubly robust check | Sensitivity analysis |
| "Who does it work for" | Uplift (causal forest / tree) | Needs a model of the effect, not the outcome | Qini + AIPW calibration |
| "Where did it move" | Contribution decomposition | Descriptive, not causal — always labelled as such | Segment anomaly scan |

The universal checklist that ships with every recommendation: identify assumptions from the design,
run placebo checks, check sign stability across specifications, test outlier dependence, and state
the scope of the claim (ATE vs ATT vs LATE). Those checks are what §6 turns into a table.

### 5.1 Open-source engine cross-validation (`ds_agent/tools/causal_oss.py`)

Home-grown implementations need independent reproduction. When a question asks for the open-source
path, the same identification problem is handed to **DoWhy** (`CausalModel` → `identify_effect` →
backdoor linear regression / PSM / IPW), optionally **EconML**'s `CausalForestDML` for a CATE
distribution, and two refutation tests run on the DoWhy estimate:

- **Placebo treatment**: replace the treatment with noise. A real effect must collapse to ≈0; a
  surviving estimate means the "effect" comes from the outcome process, not the treatment.
- **Random common cause**: add a fake confounder. The estimate must barely move; a large swing means
  it rides on specification luck.

The confidence card also reports the relative gap between the OSS estimate and the in-house PSM/IPW
estimators — agreement (<20%) across independently written engines is the strongest available
evidence that a number is not an artifact of one code path. Missing packages degrade gracefully to
the in-house estimators with an explicit notice.

### 5.2 Deep-learning anomaly detection (`ds_agent/tools/dl_anomaly.py`)

Learned-representation detectors complement the seasonal baseline:

- **Autoencoder**: a bottleneck network trained to reconstruct 7-day windows. Three choices matter
  more than the architecture: (1) the input is the **ratio to the trailing 28-day median**, not the
  raw level — otherwise trend and weekday composition dominate the reconstruction error; (2) it
  trains only on **quiet windows** (every day within 5% of the median), so it cannot memorise the
  anomaly it is supposed to catch; (3) the threshold is the **99th percentile of quiet-window
  error** with an MAD floor. Anomaly days are attributed only to deviating days inside flagged
  windows, so overlapping windows do not inflate the alert volume. With PyTorch installed an
  LSTM autoencoder replaces the MLP transparently.
- **Isolation forest**: per-day features (level, day-over-day, week-over-week, ratio to trailing
  median) fed to `IsolationForest` — the standard multivariate detector for tabular monitoring.
- **Cross-validation**: the learned detector is compared with the classical seasonal baseline by
  Jaccard overlap; the confidence card covers detection power, alert volume (≤10% of days),
  reconstruction quality and score separation.

---

## 6. Statistical tests: theory and decision rules

The full catalogue — 31 tests with their null hypotheses, formulas, assumptions, decision rules and
citations — is generated from the code into **[`statistical_tests.md`](statistical_tests.md)**
(`scripts/make_test_reference.py`). This section is the reasoning behind it: the logic every
conclusion follows, which tests each readout reports, and the theory each analysis method rests on.

### 6.1 What a conclusion must contain

A number on its own is not a conclusion. Every readout in this agent ends with the same six
elements, and the sixth is a table rather than a sentence:

| Element | Why it is there | Where it appears |
| --- | --- | --- |
| Point estimate | The answer to "how much" | KPI card / first line |
| Interval | The precision of that answer | `95% CI` next to the estimate |
| Test + statistic + p | Which hypothesis was tested and how surprising the data are under it | **test table** |
| Decision rule | p < α rejects H0; the conclusion states the reject/retain verdict, not just the number | **test table**, "conclusion" column |
| Assumption checks | A t-test on non-normal data, or a p-value from autocorrelated residuals, is a confident wrong answer | same table (normality, variance, autocorrelation, heteroskedasticity, calibration) |
| Effect size + practical reading | Significance is not importance | prose, plus "severity" / "ship or not" |

The table is produced by `ds_agent/stats/tests.py` and rendered by `to_frame`; the prose summary is
`summary_line`, which names each test that ran. A test that could not run stays in the table with a
reason (`status="skipped"`) instead of disappearing — "we did not check" and "we checked and it
passed" are different claims.

Two rules are enforced in code, not just documented:

1. **p is never reported without its test.** `TestResult` carries `name`, `h0`, `statistic` and
   `reference` together; there is no constructor for a bare p-value.
2. **A failed assumption downgrades the test, not the truth.** Normality rejection does not delete
   the t-test; it adds the rank test next to it, and the headline reports whether they agree.

### 6.2 Which tests each readout reports

| Readout | Tests | Why these |
| --- | --- | --- |
| Anomaly diagnosis | Shapiro-Wilk (or D'Agostino) on residuals, Jarque-Bera, paired t (actual vs seasonal expectation), Wilcoxon signed-rank, Ljung-Box, Durbin-Watson, ADF, KPSS | The window is compared with a *model expectation on the same days*, so the comparison is paired; residuals decide whether the normal-theory p is admissible and whether autocorrelation inflates it |
| Experiment readout | SRM chi-square, two-proportion z, chi-square independence, Fisher (sparse), D'Agostino/Shapiro per arm, Levene, Bartlett, Welch t, Student t, Mann-Whitney, KS, bootstrap, permutation, BH-FDR | SRM first, then the estimand-appropriate test, then three distribution-free cross-checks; BH keeps the metric family honest |
| DiD | Wald joint F on pre-period leads, Breusch-Pagan, Durbin-Watson (within unit, median), Jarque-Bera | The joint F is the parallel-trends assumption made testable; the residual diagnostics decide whether cluster/HAC errors are required |
| ITS / structural counterfactual | ADF, KPSS, Ljung-Box, Durbin-Watson, Jarque-Bera | Stationarity first (ADF and KPSS have opposite nulls), then residual whiteness — without it the intervention p-value is overstated |
| Modeling | Jarque-Bera, Durbin-Watson, Breusch-Pagan, VIF, Hosmer-Lemeshow, McNemar | Regression: residual assumptions and collinearity; classification: calibration and "is the winner actually better than the runner-up" |
| PSM | Paired t on matched pairs, Welch t on the worst covariate (with SMD) | The ATT gets analytic inference alongside the bootstrap; balance is judged by SMD with a test behind it |

### 6.3 Test families at a glance

| Family | Tests | Principle | Watch out | Source |
| --- | --- | --- | --- | --- |
| Normality | Shapiro-Wilk, D'Agostino K², Jarque-Bera, Anderson-Darling | Compare the empirical distribution (or its skew/kurtosis) with the normal | At n > 10⁴ trivial departures reject; read the direction, not just p | Shapiro & Wilk (1965); Jarque & Bera (1980) |
| Variance | Levene / Brown-Forsythe, Bartlett | Compare within-group dispersion of centred values | Levene degenerates when the median is 0 (mass at zero); Bartlett needs normality | Brown & Forsythe (1974) |
| Location | Student t, Welch t, paired t, Wilcoxon, Mann-Whitney, Kruskal-Wallis, ANOVA | Standardised difference in means or ranks | Welch ≠ Student when variances differ; rank tests answer a different question (stochastic dominance) | Welch (1947); Wilcoxon (1945) |
| Proportion | Two-proportion z, chi-square GOF/independence, Fisher, McNemar | Normal or exact approximation to counts | n·p < 5 breaks the z test; McNemar is for paired binary outcomes | Kohavi et al. (2020); Fisher (1935) |
| Distribution | Two-sample KS | Supremum distance between CDFs | Sensitive to any difference, not just the mean; conservative for discrete data | Massey (1951) |
| Time series | Ljung-Box, Durbin-Watson, ADF, KPSS | Residual autocorrelation; unit root vs stationarity | ADF and KPSS have opposite nulls — run both; DW is first-order only | Ljung & Box (1978); Dickey & Fuller (1979); Kwiatkowski et al. (1992) |
| Regression | Breusch-Pagan, VIF, Wald joint F, Hosmer-Lemeshow | Residual variance vs regressors; collinearity; joint restrictions; calibration | BP tests the variance, not the mean; VIF > 10 means unstable signs, not a bad fit | Breusch & Pagan (1979); Hosmer & Lemeshow (1980) |
| Resampling | Permutation, bootstrap percentile, TOST | Re-randomise or resample the assignment unit | Resample the randomisation unit, not the row; TOST needs a pre-declared margin | Fisher (1935); Efron (1979); Schuirmann (1987) |
| Multiplicity | BH-FDR, O'Brien-Fleming, mSPRT | Control the error rate across looks/metrics | FDR ≠ FWER; sequential boundaries cost power | Benjamini & Hochberg (1995); Johari et al. (2017) |

### 6.4 Assumption checks change the inference, not the narrative

The handlers do not run tests for decoration. Each check has a defined consequence:

| Check fails | Consequence in the code |
| --- | --- |
| Normality (residuals/arms) | The rank test is promoted next to the t-test and the headline reports whether they agree (`panels._panel`) |
| Equal variances | Welch is the reported test; Student t is kept in the table only to show the gap |
| Residual autocorrelation | The conclusion says the SE is optimistic; DiD/ITS already use cluster-robust or ARMA errors |
| Heteroskedasticity | Cluster/HC1 errors are the default in the causal estimators; the test documents why |
| Calibration (Hosmer-Lemeshow) | The model can still rank users, but its probabilities are not read as rates |
| Common support | The ATT is reported with the share of treated units it does not cover |

### 6.5 Theory behind each analysis method

| Method | Model / identification | Inference | Diagnostics reported | Canonical reference |
| --- | --- | --- | --- | --- |
| Seasonal baseline + CUSUM (§1) | `y_t = level_t · seasonal_dow(t) · (1 + ε_t)`, causal trailing level with growth correction | Robust z from MAD, CUSUM with slack 0.5 / threshold 5, one-sample t on the run | Episode t-test, leave-one-out recheck, MASE of the baseline | Page (1954); Cleveland et al. (1990) |
| Holt-Winters / SARIMA / STL+ETS (§1.5) | Exponential smoothing with damped trend and weekly season; SARIMAX(1,1,1)(0,1,1)₇ | Rolling-origin one-step errors; MASE against seasonal naive | Backtest table, selection margin, residual Ljung-Box | Hyndman & Athanasopoulos (2021) |
| Contribution / attribution (§2) | Additive: `Δ = Σ Δv_i`; ratio: rate + mix + interaction, exact by construction | Descriptive (no sampling model) — labelled as such | Share of change, concentration, partial-answer disclosure | — (arithmetic identity) |
| Power & sample size (§3.1) | `power = Φ(δ/SE − z) + Φ(−δ/SE − z)`, one solver for every metric family | Brent inversion of the same power function | MDE curve, duration, ICC inflation | Kohavi et al. (2020), Ch. 3 |
| AB readout (§3.2) | Randomisation ⇒ exchangeability; CUPED `y − θ(x − x̄)` | Two-proportion z / Welch t / delta method; BH across metrics; OBF or mSPRT for peeking | SRM, permutation cross-check, segment heterogeneity | Deng et al. (2013); Johari et al. (2017) |
| Switchback (§3.4) | Time block is the inference unit; `y_t = α + τ·arm_t + trend + DOW + ε_t` | Newey-West HAC, lag ≈ 1; paired-block difference | Washout, carryover coefficient, residual autocorrelation | Bojinov et al. (2023) |
| Reversal (§3.4) | Arms swap at a midpoint; a real effect flips sign | τ₁, τ₂, sum test τ₁+τ₂≈0, persistence | Infrastructure validity, persistence ratio | — (design-based) |
| Parent-child (§3.4) | Orthogonal layers; `y = τ·child + α_parent + ε` | HC1 with parent fixed effects; interaction term | Chi-square orthogonality, interaction CI | Kohavi et al. (2020), Ch. 7 |
| Two-sided 2×2 (§3.4) | Four cells (00,10,01,11); direct effects from marginals | Market-level bootstrap | Interaction `(11−10)−(01−00)`, spillover ratio | — (design-based) |
| DiD / event study (§4.1–4.2) | `y_it = β(treated_i × post_t) + α_i + γ_t + ε_it`; parallel trends | Cluster-robust SE; joint F on leads; in-space placebo | Pre-trend F, placebo distribution, level vs log | Goodman-Bacon (2021); Angrist & Pischke (2009) |
| PSM / IPW / AIPW (§4.3) | `e(x) = P(T=1|x)`; conditional ignorability + overlap | ATT/ATE with bootstrap CI; Hájek weights; doubly robust scores | SMD, paired t, common support, estimator agreement | Rosenbaum & Rubin (1983); Imbens & Rubin (2015) |
| Synthetic control (§4.4) | `w ≥ 0, Σw = 1` minimising pre-period RMSE | In-space placebo rank `p = rank/(n+1)` | Donor weights, pre-period fit, placebo gap ratio | Abadie et al. (2010) |
| RDD (§4.6) | Treatment switches at a cutoff; local randomisation | Local linear, triangular kernel, HC1; CCT bandwidth | Density manipulation, covariate jumps, placebo cutoffs, bandwidth stability | Lee & Lemieux (2010); McCrary (2008) |
| ITS / causal impact (§4.7) | `f(y_t) = β·post + γ₁trend + γ₂DOW + u_t`, `u ~ ARMA`; state-space counterfactual | SARIMAX with estimated error structure; state-space variance widening with horizon | Ljung-Box, placebo dates, pre-period R², specification curve | Brodersen et al. (2015) |
| Uplift (§4.5) | `τ(x) = E[Y(1)−Y(0)|X=x]`; honest splits | Causal forest CATE, Qini, cross-fitted AIPW | Qini curve, decile capture, learner comparison | Athey & Imbens (2016); Athey et al. (2019); Künzel et al. (2019) |
| DL anomaly (§5.2) | Autoencoder reconstruction of 7-day windows; isolation forest | Reconstruction error vs quiet-window quantile; Jaccard overlap with the seasonal detector | Alert volume, reconstruction quality, score separation | — (implementation-defined) |
| Sampling (§7) | Sampling distribution of the mean; `n = (z_{1−α/2}+z_{1−β})²·σ²/δ²` (or `p₀(1−p₀)`) | z-based sizing; chi-square GOF / Welch t / KS for representativeness | SE + 95% CI on every draw, stratum allocation, share-vs-population table | Cochran (1977); Massey (1951) |
| Modeling (§5.3) | Four-stage pipeline shared with causal inference: preprocess (missing/NaN) → feature engineering (construction + collinearity pruning) → linear vs gradient-boosted model → post-validation feature analysis | Held-out metrics, 5-fold CV, calibration, McNemar between models | Missing/NaN profile, VIF + pairwise-|r| pruning log, BP, DW, JB, HL | Hastie et al. (2009) |

### 6.6 Reading the table in an answer

A row such as `Welch t-test | t=-3.12 | p=0.003 | the group means differ significantly` is three
claims in one line: which hypothesis, how surprising the data are, and what that licenses you to say.
Two rows that disagree (e.g. Welch rejects, Mann-Whitney does not) are not a bug — they are the
answer to "is the conclusion robust to the distributional assumption", and the headline says so.
This is why `panels._panel` reports an `agreement` flag instead of averaging p-values.

---

## 7. Sampling: theory and tools (`ds_agent/stats/sampling.py`)

Sampling is where the CLT stops being a theorem and starts being a budget. The
module implements the three questions a practitioner actually asks, and every
agent answer carries the formula and the reference, not just the number.

### 7.1 The sampling distribution (the theory every number leans on)

For i.i.d. observations the sample mean is approximately normal with mean μ
and **standard error** SE = s/√n (the finite-population form multiplies by
√((N−n)/(N−1)), negligible when n ≪ N). Three consequences the agent states
explicitly:

- a 95% CI is mean ± t(1−α/2, n−1)·SE — reported on every sample draw;
- precision is expensive: halving the detectable effect δ needs **4× the
  sample**, because n ∝ 1/δ²;
- representativeness is testable: the sample should look like the sampling
  distribution's parent, and the tests below read off that claim.

### 7.2 Sample size determination

- **Proportions**: `n = (z_{1−α/2} + z_{1−β})² · p₀(1−p₀) / δ²` (two-sided,
  normal approximation; a note recommends exact binomial sizing when p₀ < 1%).
- **Means**: `n = (z_{1−α/2} + z_{1−β})² · σ² / δ²`.
- The answer ships z-values, the formula in KaTeX-ish text, a sensitivity
  table (n at δ/2 … 2δ), and Cochran (1977) as the citation. This is the same
  power equation the experiment-design readout uses — sizing a survey and
  sizing an experiment are the same arithmetic.

### 7.3 Drawing a sample

`draw_sample` implements, with a fixed seed so any draw is reproducible and
auditable:

- **simple random** — uniform without replacement (`default_rng.choice`);
- **systematic** — every k-th row after a random start, k = N/n (fine for
  ordering-free frames; dangerous when rows are seasonally ordered — noted);
- **stratified** — proportional allocation across a grouping column with SRS
  inside each stratum and a ≥1 floor, so rare-but-decision-relevant segments
  keep representation; the allocation table ships with the draw.

Where sampling decisions bite in internet data work:

- **Cluster sampling** — when treatment applies to a cluster (city, market),
  the cluster is the sampling *and* inference unit; SEs come from cluster
  bootstrap. Treating clustered rows as independent is the most common
  variance bug in marketplace analysis.
- **Time-slice sampling** — switchback experiments sample time blocks; the
  block, not the user, carries the randomisation, so block-level HAC errors
  are mandatory.
- **Bootstrap** — resample with replacement at the correct unit for CIs
  without distributional assumptions (PSM, two-sided markets).
- **Sample-ratio mismatch (SRM)** — not sampling for analysis but a trust
  check: the realised split must match the planned split.
- **CUPED** — variance reduction from pre-experiment data is free sampling
  efficiency: same n, smaller variance, shorter tests.

### 7.4 Representativeness testing

`representativeness` compares sample vs population with the proper test
instead of eyeballing means:

- **categorical** — chi-square goodness-of-fit with the population proportions
  as expected shares; the answer includes a per-category share table
  (sample % vs population % vs difference);
- **numeric** — Welch t on the means (same average?) plus a two-sample KS on
  the full distribution (same shape?), so a representative-looking mean cannot
  hide a skewed draw.

All three tests come from the shared catalogue in
[`statistical_tests.md`](statistical_tests.md), so the sampling readout gets
the same H0/formula/decision/citation table as every other readout.

---

## 8. Business-knowledge retrieval (`ds_agent/knowledge.py`)

Numbers without calibers mislead. The agent carries a small internal knowledge base
(`docs/business_knowledge.md`: metric calibers, dimension semantics, data dictionary, known
incidents) and retrieves from it with a **LangChain + LangGraph** RAG (pure-Python fallback keeps
the same behaviour without the frameworks):

- **Chunking** uses LangChain's `MarkdownHeaderTextSplitter`: each `##`/`###` heading becomes one
  chunk whose title inherits its parent ("指标口径 > GMV"), so a chunk is a self-contained knowledge unit.
- **Retrieval** is a custom LangChain `BaseRetriever`: TF-IDF over Chinese character bigrams + ASCII
  words, with title and body scored as separate length-normalised fields (2:1) and IDF weighting —
  deterministic, offline, and accurate enough for a corpus of dozens of chunks where an embedding
  service would be overkill.
- **Orchestration** is a LangGraph `StateGraph` with three nodes — `retrieve` (top-k chunks),
  `grade` (relevance threshold, re-checkable without touching retrieval) and `compose` (prompt-ready
  context block). `KnowledgeBase.answer()` runs the compiled graph and reports the engine used.
- **Wiring.** SQL answers append the relevant caliber note (so "查 GMV" also reminds you GMV includes
  unpaid orders and ignores refunds); reports open with a business-context section; and direct
  questions ("GMV 的口径是什么") are answered from the top-k chunks with sources and relevance.

---

## 8b. Confidence validation (`ds_agent/confidence.py`) and root-cause trees (`ds_agent/tools/root_cause.py`)

A point estimate without a trust assessment is an anecdote. Since this round, the five
first-class readouts (anomaly, AB, DiD, PSM, time-series causal) end with a standardized
confidence card:

- each check is binary or three-state (PASS / WARN / FAIL) with a bilingual explanation;
- the verdict aggregates to 高置信 / 中等置信 / 低置信 — any FAIL caps the answer at 低置信,
  because a broken randomisation or violated pre-trend invalidates everything downstream;
- the validation ships a chart, not just text: a ±2.5σ noise band for anomalies, the permutation
  null histogram for AB tests, and the in-space placebo histogram for DiD. The placebo and
  permutation procedures are seeded, so every confidence card is reproducible.

The DiD in-space placebo permutes the treated label across units (same treated count) and re-runs
TWFE; the reported ATT should exceed ~95% of the placebo effects. The AB permutation test shuffles
arm labels for the primary metric (exact for proportion/mean estimands, deliberately skipped for
ratio metrics where reassignment would change the estimand).

Root-cause analysis goes beyond one-level contribution decomposition with a recursive drill-down
tree: at each node the remaining dimensions are scored by "largest segment contribution ×
concentration", the winner branches into its top segments above a contribution threshold
(default 8% of the node's move), up to 3 levels and a node budget. Leaves ranked by share of the
root move are the root-cause shortlist; because every node reuses the exact decompositions of
§"Contribution", path numbers stay additive and reconcilable with the topline.

---

## 9. Reporting (`ds_agent/tools/report.py`)

A report is a list of `Section`s (heading, prose, table, chart). `to_markdown()` emits a clean
Markdown file; `to_html()` inlines CSS and base64 PNGs so the file can be forwarded as a single
attachment. Headings are bilingual and the structure is conclusion-first:

1. Executive summary (the finding, the driver, the significance)
2. Evidence (charts and tables)
3. Method and definitions (so a reader can reproduce it)
4. Caveats and next steps

---

## 10. References

- Cleveland, R. B. et al. (1990). *STL: A seasonal-trend decomposition procedure based on loess.*
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.) — Holt-Winters
  and MASE.
- Page, E. S. (1954). *Continuous inspection schemes.* (CUSUM)
- Benjamini, Y. & Hochberg, Y. (1995). *Controlling the false discovery rate.*
- Deng, A., Xu, Y., Kohavi, R., Walker, T. (2013). *Improving the sensitivity of online controlled
  experiments using CUPED.*
- Johari, R., Koomen, P., Pekelis, L., Walsh, D. (2017). *Peeking at A/B tests.* (mSPRT)
- Kohavi, R., Tang, D., Xu, Y. (2020). *Trustworthy Online Controlled Experiments* — SRM, guardrails,
  and the analysis patterns behind the readout.
- Bojinov, I., Simchi-Levi, D., Zhao, J. (2023). *Design and analysis of switchback experiments.*
- Athey, S., Imbens, G., Wager, S. (2019). *Estimating treatment effects with causal forests.*
- Athey, S. & Imbens, G. (2016). *Recursive partitioning for heterogeneous causal effects.* (causal tree)
- Künzel, S. et al. (2019). *Metalearners for estimating heterogeneous treatment effects.* (T/S/X)
- Hahn, J., Todd, P., Van der Klaauw, W. (2001). *Identification and estimation of treatment effects
  with a regression-discontinuity design.*
- Lee, D. & Lemieux, T. (2010). *Regression discontinuity designs in economics.*
- McCrary, J. (2008). *Manipulation of the running variable in the regression discontinuity design.*
- Brodersen, K. et al. (2015). *Inferring causal impact using Bayesian structural time-series models.*
- Goodman-Bacon, A. (2021). *Difference-in-differences with variation in treatment timing.*
- Abadie, A., Diamond, A., Hainmueller, J. (2010). *Synthetic control methods.*
- Rosenbaum, P. & Rubin, D. (1983). *The central role of the propensity score.*
- Imbens, G. & Rubin, D. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences* —
  potential outcomes, ATT vs ATE, and the overlap condition used by the matching diagnostics.
- Benjamini, Y. & Hochberg, Y. (1995) aside, see also Efron, B. (1979). *Bootstrap methods* — the
  inference unit must match the assignment unit.

Each statistical test also carries its own citation in
[`statistical_tests.md`](statistical_tests.md); the ones used most often across the readouts:

- Welch, B. L. (1947). *The generalization of Student's problem when several different population
  variances are involved.* Biometrika 34(1/2), 28-35.
- Wilcoxon, F. (1945). *Individual comparisons by ranking methods.* Biometrics Bulletin 1(6), 80-83.
- Brown, M. B. & Forsythe, A. B. (1974). *Robust tests for the equality of variances.* JASA 69(346).
- Jarque, C. M. & Bera, A. K. (1980). *Efficient tests for normality, homoscedasticity and serial
  independence of regression residuals.* Economics Letters 6(3), 255-259.
- Ljung, G. M. & Box, G. E. P. (1978). *On a measure of lack of fit in time series models.*
  Biometrika 65(2), 297-303.
- Dickey, D. A. & Fuller, W. A. (1979). *Distribution of the estimators for autoregressive time
  series with a unit root.* JASA 74(366), 427-431.
- Kwiatkowski, D. et al. (1992). *Testing the null hypothesis of stationarity.* Journal of
  Econometrics 54(1-3), 159-178.
- Breusch, T. S. & Pagan, A. R. (1979). *A simple test for heteroscedasticity.* Econometrica 47(5).
- Hosmer, D. W. & Lemesbow, S. (1980). *Goodness of fit tests for the multiple logistic regression
  model.* Communications in Statistics A9(10), 1043-1069.
- McNemar, Q. (1947). *Note on the sampling error of the difference between correlated proportions.*
  Psychometrika 12(2), 153-157.
- Schuirmann, D. J. (1987). *A comparison of the two one-sided tests procedure and the power
  approach.* Journal of Pharmacokinetics and Biopharmaceutics 15(6), 657-680.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.).
  Springer — the modeling and feature-analysis stack.
- Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.). Wiley — sample size formulas and the
  stratified/systematic designs behind the sampling toolkit.
- Massey, F. J. (1951). *The Kolmogorov-Smirnov test for goodness of fit.* JASA 46(253).

