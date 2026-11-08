from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from .model import ReactionNetwork

@dataclass
class SimulationResult:
    time: np.ndarray
    trajectories: Dict[str, np.ndarray]
    reaction_counts: np.ndarray

    def get_species_trajectory(self, species_name: str) -> np.ndarray:
        return self.trajectories[species_name]

class GillespieSSA:
    def __init__(self, network: ReactionNetwork, seed: Optional[int] = None):
        self.network = network
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        initial_state: Dict[str, int],
        t_max: float,
        max_steps: int = 1_000_000,
    ) -> SimulationResult:
        state = initial_state.copy()
        # Ensure all species are present in state
        for s in self.network.species:
            if s not in state:
                state[s] = 0

        t = 0.0
        time_history = [t]
        state_history = {s: [state[s]] for s in self.network.species}
        reaction_counts = np.zeros(len(self.network.reactions), dtype=np.int64)

        step = 0
        while t < t_max and step < max_steps:
            propensities = self.network.get_propensities(state)
            a_0 = np.sum(propensities)

            if a_0 <= 0.0:
                # No further reactions can occur
                break

            # Sample time increment from exponential distribution
            r1 = self.rng.random()
            while r1 == 0.0:
                r1 = self.rng.random()
            tau = -np.log(r1) / a_0

            if t + tau > t_max:
                # End reached
                t = t_max
                time_history.append(t)
                for s in self.network.species:
                    state_history[s].append(state[s])
                break

            # Sample reaction index based on discrete propensity distribution
            r2 = self.rng.random() * a_0
            cum_propensities = np.cumsum(propensities)
            rxn_idx = int(np.searchsorted(cum_propensities, r2))
            if rxn_idx >= len(self.network.reactions):
                rxn_idx = len(self.network.reactions) - 1

            # Apply reaction state update
            rxn = self.network.reactions[rxn_idx]
            for s, count in rxn.reactants.items():
                state[s] -= count
            for s, count in rxn.products.items():
                state[s] += count

            reaction_counts[rxn_idx] += 1
            t += tau
            step += 1

            time_history.append(t)
            for s in self.network.species:
                state_history[s].append(state[s])

        return SimulationResult(
            time=np.array(time_history, dtype=np.float64),
            trajectories={s: np.array(vals, dtype=np.int64) for s, vals in state_history.items()},
            reaction_counts=reaction_counts,
        )

class TauLeaping:
    def __init__(self, network: ReactionNetwork, epsilon: float = 0.03, seed: Optional[int] = None):
        self.network = network
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        initial_state: Dict[str, int],
        t_max: float,
        dt: Optional[float] = None,
        max_steps: int = 500_000,
    ) -> SimulationResult:
        state = initial_state.copy()
        for s in self.network.species:
            if s not in state:
                state[s] = 0

        t = 0.0
        time_history = [t]
        state_history = {s: [state[s]] for s in self.network.species}
        reaction_counts = np.zeros(len(self.network.reactions), dtype=np.int64)

        step = 0
        while t < t_max and step < max_steps:
            propensities = self.network.get_propensities(state)
            a_0 = np.sum(propensities)
            if a_0 <= 0.0:
                break

            if dt is None:
                # Compute adaptive tau leap step
                # Simple Cao et al. approximation
                tau = min(0.1, 1.0 / (a_0 + 1e-9))
                tau = max(1e-4, tau)
            else:
                tau = dt

            if t + tau > t_max:
                tau = t_max - t

            # Sample number of firings for each reaction using Poisson
            firings = self.rng.poisson(propensities * tau)

            # Check if any species count would drop below zero
            tentative_state = state.copy()
            valid = True
            for r_idx, k in enumerate(firings):
                if k > 0:
                    rxn = self.network.reactions[r_idx]
                    for s, count in rxn.reactants.items():
                        tentative_state[s] -= count * k
                        if tentative_state[s] < 0:
                            valid = False
                            break
                    if not valid:
                        break

            if not valid or tau < 1e-5:
                # Fallback to single exact SSA step to prevent negative population
                r1 = self.rng.random()
                while r1 == 0.0:
                    r1 = self.rng.random()
                tau_exact = -np.log(r1) / a_0
                r2 = self.rng.random() * a_0
                cum_prop = np.cumsum(propensities)
                rxn_idx = int(np.searchsorted(cum_prop, r2))
                if rxn_idx >= len(self.network.reactions):
                    rxn_idx = len(self.network.reactions) - 1

                rxn = self.network.reactions[rxn_idx]
                for s, count in rxn.reactants.items():
                    state[s] -= count
                for s, count in rxn.products.items():
                    state[s] += count

                reaction_counts[rxn_idx] += 1
                t += tau_exact
            else:
                for r_idx, k in enumerate(firings):
                    if k > 0:
                        rxn = self.network.reactions[r_idx]
                        for s, count in rxn.reactants.items():
                            state[s] -= count * k
                        for s, count in rxn.products.items():
                            state[s] += count * k
                        reaction_counts[r_idx] += k
                t += tau

            step += 1
            time_history.append(t)
            for s in self.network.species:
                state_history[s].append(state[s])

        return SimulationResult(
            time=np.array(time_history, dtype=np.float64),
            trajectories={s: np.array(vals, dtype=np.int64) for s, vals in state_history.items()},
            reaction_counts=reaction_counts,
        )
