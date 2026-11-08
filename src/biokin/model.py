from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

@dataclass
class Species:
    name: str
    initial_count: int = 0

@dataclass
class Reaction:
    name: str
    reactants: Dict[str, int]
    products: Dict[str, int]
    rate_constant: float
    propensity_fn: Optional[Callable[[Dict[str, int]], float]] = None

    def calculate_propensity(self, state: Dict[str, int]) -> float:
        if self.propensity_fn is not None:
            return float(self.propensity_fn(state))
        
        # Standard mass-action propensity calculation
        h = 1.0
        for species, stoich in self.reactants.items():
            count = state.get(species, 0)
            if stoich == 1:
                h *= count
            elif stoich == 2:
                h *= count * (count - 1) / 2.0
            elif stoich == 3:
                h *= count * (count - 1) * (count - 2) / 6.0
            else:
                # General combination for higher-order stoichiometry
                import math
                if count < stoich:
                    return 0.0
                h *= math.comb(count, stoich)
        
        return float(self.rate_constant * h)

@dataclass
class ReactionNetwork:
    species: List[str] = field(default_factory=list)
    reactions: List[Reaction] = field(default_factory=list)
    _species_to_idx: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self._rebuild_indices()

    def _rebuild_indices(self):
        self._species_to_idx = {name: idx for idx, name in enumerate(self.species)}

    def add_species(self, name: str) -> None:
        if name not in self._species_to_idx:
            self.species.append(name)
            self._rebuild_indices()

    def add_reaction(self, reaction: Reaction) -> None:
        for s in list(reaction.reactants.keys()) + list(reaction.products.keys()):
            self.add_species(s)
        self.reactions.append(reaction)

    @property
    def stoichiometry_matrix(self) -> np.ndarray:
        num_species = len(self.species)
        num_reactions = len(self.reactions)
        stoich = np.zeros((num_species, num_reactions), dtype=np.int64)
        
        for r_idx, rxn in enumerate(self.reactions):
            for s_name, count in rxn.reactants.items():
                s_idx = self._species_to_idx[s_name]
                stoich[s_idx, r_idx] -= count
            for s_name, count in rxn.products.items():
                s_idx = self._species_to_idx[s_name]
                stoich[s_idx, r_idx] += count
                
        return stoich

    def get_propensities(self, state: Dict[str, int]) -> np.ndarray:
        return np.array([rxn.calculate_propensity(state) for rxn in self.reactions], dtype=np.float64)
