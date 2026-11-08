# BioKin — Gillespie Stochastic Simulation Engine for Gene Regulatory Networks

[![CI](https://github.com/Sunzzx/biokin-gillespie-stochastic-gene-network/actions/workflows/ci.yml/badge.svg)](https://github.com/Sunzzx/biokin-gillespie-stochastic-gene-network/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=github-sponsors)](https://github.com/sponsors/Sunzzx)
[![Polar](https://img.shields.io/badge/Polar-Support%20this%20project-blue)](https://polar.sh/sunzzx)

**Exact Gillespie Direct Method (SSA), Adaptive Tau-Leaping, and synthetic biology circuit library** for stochastic chemical kinetics and gene regulatory network simulation.

BioKin implements the classic **Doob–Gillespie Stochastic Simulation Algorithm** from first principles, enabling exact (in distribution) Monte Carlo trajectories of continuous-time Markov jump processes that describe molecular counts inside living cells.

---

## Humanity & Real-World Impact

Intracellular chemistry is *noisy*. When molecule counts are low (tens to hundreds of transcription factors, mRNAs, or metabolites), continuum ODE approximations fail and rare stochastic events dominate phenotype:

- **Synthetic biology & genetic circuit design** — Predict whether a repressilator will oscillate, a toggle switch will flip, or a feed-forward loop will filter noise *before* wet-lab construction, cutting experimental cost and time.
- **Systems medicine** — Model stochastic gene expression bursts underlying drug resistance, cell-fate decisions, and pathogen latency (HIV, tuberculosis).
- **Pharmacokinetics / toxicology** — Simulate sparse binding events and receptor occupancy under low ligand concentrations.
- **Education & open science** — Provide a zero-friction, pure-Python reference implementation of SSA / tau-leaping so students and labs worldwide can explore stochastic kinetics without commercial licenses.

By making exact stochastic simulation accessible, BioKin lowers the barrier between theoretical systems biology and practical therapeutic / biotech design.

---

## Theoretical Foundations

### Chemical Master Equation (CME)

Let $\mathbf{X}(t) \in \mathbb{N}^{N}$ be the vector of molecular counts for $N$ species. The Chemical Master Equation governs the probability $P(\mathbf{x}, t)$ of being in state $\mathbf{x}$ at time $t$:

$$
\frac{\partial P(\mathbf{x}, t)}{\partial t}
=
\sum_{j=1}^{M}
\Bigl[
a_j(\mathbf{x} - \boldsymbol{\nu}_j)\, P(\mathbf{x} - \boldsymbol{\nu}_j, t)
-
a_j(\mathbf{x})\, P(\mathbf{x}, t)
\Bigr]
$$

where $a_j(\mathbf{x})$ is the **propensity** of reaction $j$ and $\boldsymbol{\nu}_j \in \mathbb{Z}^{N}$ is its stoichiometry vector.

### Gillespie Direct Method (SSA)

The waiting time $\tau$ until the next reaction is exponentially distributed:

$$
P(\tau \ge t) = e^{-a_0(\mathbf{x})\, t},
\qquad
a_0(\mathbf{x}) = \sum_{j=1}^{M} a_j(\mathbf{x})
$$

Sample:

$$
\tau = \frac{1}{a_0} \ln\Bigl(\frac{1}{r_1}\Bigr),
\qquad
r_1 \sim U(0,1)
$$

Select reaction $j$ with probability $a_j / a_0$:

$$
\sum_{k=1}^{j-1} a_k < r_2 a_0 \le \sum_{k=1}^{j} a_k,
\qquad
r_2 \sim U(0,1)
$$

Update state: $\mathbf{x} \leftarrow \mathbf{x} + \boldsymbol{\nu}_j$.

### Mass-Action Propensities

For a reaction with stoichiometry $\mathbf{s}$ (reactant counts) and rate constant $c$:

$$
a(\mathbf{x}) = c \cdot h(\mathbf{x}),
\qquad
h(\mathbf{x}) = \prod_i \binom{x_i}{s_i}
$$

Special cases:

- Unimolecular $A \to \ldots$: $h = x_A$
- Bimolecular $2A \to \ldots$: $h = x_A(x_A-1)/2$
- Heterogeneous $A + B \to \ldots$: $h = x_A x_B$

### Adaptive Tau-Leaping (Cao et al.)

When populations are large, leap over many reactions by Poisson sampling:

$$
K_j(\tau) \sim \mathrm{Poisson}\bigl(a_j(\mathbf{x})\,\tau\bigr)
$$

with leap size $\tau$ chosen so relative change in each propensity stays within tolerance $\varepsilon$:

$$
\tau = \min_i \frac{\max\{\varepsilon x_i / g_i,\, 1\}}{\bigl|\sum_j \nu_{ij} a_j\bigr|}
$$

BioKin falls back to exact SSA steps whenever a leap would drive any species negative.

### Built-in Gene Circuits

| Circuit | Biological Role | Key Dynamics |
|---------|-----------------|--------------|
| **Repressilator** (Elowitz & Leibler, 2000) | Synthetic oscillator | Limit-cycle oscillations via cyclic repression |
| **Genetic Toggle Switch** (Gardner et al., 2000) | Bistable memory | Mutual repression → two stable attractors |
| **Coherent Type-1 FFL** | Noise filter / delay | AND-gate logic: output Z requires both X and Y |

Hill-type repression propensity used in circuits:

$$
a_{\mathrm{tx}} = \alpha_0 + \frac{\alpha}{1 + (p / K)^n}
$$

---

## Complexity Analysis

| Algorithm | Time per step | Space | Notes |
|-----------|---------------|-------|-------|
| Gillespie Direct SSA | $O(M)$ propensity + $O(M)$ selection | $O(N + M)$ | Exact; step count $\propto a_0 T$ |
| Tau-Leaping | $O(M)$ per leap | $O(N + M)$ | Approximate; fewer steps when $a_0$ large |
| Stoichiometry matrix build | $O(N M)$ | $O(N M)$ | Sparse in practice |

For $T$ simulation time with mean propensity scale $\bar{a}_0$, expected SSA steps $\sim \bar{a}_0 T$.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    BioKin Engine                        │
├──────────────┬──────────────────┬───────────────────────┤
│   model.py   │  simulation.py   │     circuits.py       │
│              │                  │                       │
│ Species      │ GillespieSSA     │ Repressilator         │
│ Reaction     │   · Direct Method│ GeneticToggleSwitch   │
│ ReactionNet  │   · Exact τ, μ   │ FeedForwardLoop       │
│ · Stoich S   │ TauLeaping       │ · Hill propensities   │
│ · Propensity │   · Poisson leap │ · Canonical ICs       │
│              │   · SSA fallback │                       │
└──────┬───────┴────────┬─────────┴───────────┬───────────┘
       │                │                     │
       └────────────────┼─────────────────────┘
                        ▼
              SimulationResult
              · time[]  trajectories{}  reaction_counts[]
                        │
                        ▼
              demo.py  (ASCII live plots)
```

---

## Installation

```bash
# From source
git clone https://github.com/Sunzzx/biokin-gillespie-stochastic-gene-network.git
cd biokin-gillespie-stochastic-gene-network
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

---

## Quickstart

```python
from biokin import GillespieSSA, Repressilator

network, initial = Repressilator.build_network()
ssa = GillespieSSA(network, seed=42)
result = ssa.run(initial, t_max=200.0)

print(result.time[-1], result.trajectories["p_tetr"][-1])
```

### Interactive Demo

```bash
uv run python -m biokin.demo --circuit all
# or
uv run biokin-demo --circuit repressilator
```

### Run Tests

```bash
uv run pytest -q
```

---

## Benchmark Snapshot

```
Birth-death (λ=5, μ=0.5, T=200, seed=123):
  SSA steady-state mean A ≈ 10.1  (theory: λ/μ = 10)

Repressilator (T=200, seed=42):
  ~10⁴–10⁵ SSA steps; three protein trajectories exhibit out-of-phase oscillations

Tau-leaping vs SSA (birth-death λ=10, μ=0.1, T=50):
  SSA mean ≈ 99–101 | Tau mean ≈ 95–105  (ε≈0.03)
```

---

## Monetization & Support

- **GitHub Sponsors:** [github.com/sponsors/Sunzzx](https://github.com/sponsors/Sunzzx)
- **Polar.sh:** [polar.sh/sunzzx](https://polar.sh/sunzzx)
- **Commercial / Enterprise licensing & custom circuit consulting:** ayushkandwal2007@gmail.com
- **Stripe commercial license inquiries:** contact operator for private checkout link

---

## Citation

If you use BioKin in research, please cite:

```bibtex
@software{biokin2026,
  author = {Kandwal, Ayush},
  title  = {BioKin: Gillespie Stochastic Simulation Engine for Gene Regulatory Networks},
  year   = {2026},
  url    = {https://github.com/Sunzzx/biokin-gillespie-stochastic-gene-network}
}
```

Primary references:

- Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical reactions. *J. Phys. Chem.*
- Cao, Y., Gillespie, D. T., & Petzold, L. R. (2006). Efficient step size selection for the tau-leaping simulation method. *J. Chem. Phys.*
- Elowitz, M. B., & Leibler, S. (2000). A synthetic oscillatory network of transcriptional regulators. *Nature.*
- Gardner, T. S., Cantor, C. R., & Collins, J. J. (2000). Construction of a genetic toggle switch in *Escherichia coli*. *Nature.*

---

## License

MIT © 2026 Ayush Kandwal
