from __future__ import annotations
from typing import Dict, Tuple
from .model import Reaction, ReactionNetwork

class Repressilator:
    """
    Synthetic oscillating gene network (Elowitz & Leibler, Nature 2000).
    TetR inhibits cI, cI inhibits LacI, LacI inhibits TetR.
    """
    @staticmethod
    def build_network(
        alpha0: float = 0.5,
        alpha: float = 50.0,
        n: float = 2.0,
        kd: float = 0.1,
        km: float = 0.2,
        kp: float = 1.0,
    ) -> Tuple[ReactionNetwork, Dict[str, int]]:
        network = ReactionNetwork()
        species = ["m_tetr", "m_ci", "m_laci", "p_tetr", "p_ci", "p_laci"]
        for s in species:
            network.add_species(s)

        # Transcription reactions with Hill-type repression
        def prop_tetr(state: Dict[str, int]) -> float:
            p_l = state.get("p_laci", 0)
            return alpha0 + alpha / (1.0 + (p_l / 10.0) ** n)

        def prop_ci(state: Dict[str, int]) -> float:
            p_t = state.get("p_tetr", 0)
            return alpha0 + alpha / (1.0 + (p_t / 10.0) ** n)

        def prop_laci(state: Dict[str, int]) -> float:
            p_c = state.get("p_ci", 0)
            return alpha0 + alpha / (1.0 + (p_c / 10.0) ** n)

        network.add_reaction(Reaction("tx_tetr", {}, {"m_tetr": 1}, 1.0, propensity_fn=prop_tetr))
        network.add_reaction(Reaction("tx_ci", {}, {"m_ci": 1}, 1.0, propensity_fn=prop_ci))
        network.add_reaction(Reaction("tx_laci", {}, {"m_laci": 1}, 1.0, propensity_fn=prop_laci))

        # Translation reactions
        network.add_reaction(Reaction("tl_tetr", {"m_tetr": 1}, {"m_tetr": 1, "p_tetr": 1}, kp))
        network.add_reaction(Reaction("tl_ci", {"m_ci": 1}, {"m_ci": 1, "p_ci": 1}, kp))
        network.add_reaction(Reaction("tl_laci", {"m_laci": 1}, {"m_laci": 1, "p_laci": 1}, kp))

        # mRNA degradation
        network.add_reaction(Reaction("deg_m_tetr", {"m_tetr": 1}, {}, km))
        network.add_reaction(Reaction("deg_m_ci", {"m_ci": 1}, {}, km))
        network.add_reaction(Reaction("deg_m_laci", {"m_laci": 1}, {}, km))

        # Protein degradation
        network.add_reaction(Reaction("deg_p_tetr", {"p_tetr": 1}, {}, kd))
        network.add_reaction(Reaction("deg_p_ci", {"p_ci": 1}, {}, kd))
        network.add_reaction(Reaction("deg_p_laci", {"p_laci": 1}, {}, kd))

        initial_state = {
            "m_tetr": 0, "p_tetr": 10,
            "m_ci": 0, "p_ci": 0,
            "m_laci": 0, "p_laci": 0,
        }
        return network, initial_state

class GeneticToggleSwitch:
    """
    Gardner, Cantor & Collins (Nature 2000) Bistable Genetic Toggle Switch.
    Repressor 1 mutually represses Repressor 2.
    """
    @staticmethod
    def build_network(
        alpha1: float = 30.0,
        alpha2: float = 30.0,
        beta: float = 2.0,
        gamma: float = 1.0,
    ) -> Tuple[ReactionNetwork, Dict[str, int]]:
        network = ReactionNetwork()
        network.add_species("u")  # Repressor 1
        network.add_species("v")  # Repressor 2

        def prop_synth_u(state: Dict[str, int]) -> float:
            v = state.get("v", 0)
            return alpha1 / (1.0 + (v / 10.0) ** beta)

        def prop_synth_v(state: Dict[str, int]) -> float:
            u = state.get("u", 0)
            return alpha2 / (1.0 + (u / 10.0) ** beta)

        network.add_reaction(Reaction("synth_u", {}, {"u": 1}, 1.0, propensity_fn=prop_synth_u))
        network.add_reaction(Reaction("synth_v", {}, {"v": 1}, 1.0, propensity_fn=prop_synth_v))
        network.add_reaction(Reaction("deg_u", {"u": 1}, {}, gamma))
        network.add_reaction(Reaction("deg_v", {"v": 1}, {}, gamma))

        initial_state = {"u": 20, "v": 0}
        return network, initial_state

class FeedForwardLoop:
    """
    Coherent Feed-Forward Loop (Type 1 FFL) with AND-gate activation.
    Input X activates Y and Z; Y activates Z.
    """
    @staticmethod
    def build_network(
        k_x: float = 1.0,
        deg_x: float = 0.1,
        k_y: float = 0.5,
        deg_y: float = 0.1,
        k_z: float = 0.5,
        deg_z: float = 0.1,
    ) -> Tuple[ReactionNetwork, Dict[str, int]]:
        network = ReactionNetwork()
        for s in ["x", "y", "z"]:
            network.add_species(s)

        def prop_synth_y(state: Dict[str, int]) -> float:
            x = state.get("x", 0)
            return k_y * (x / (5.0 + x))

        def prop_synth_z(state: Dict[str, int]) -> float:
            x = state.get("x", 0)
            y = state.get("y", 0)
            # AND-gate activation
            return k_z * (x / (5.0 + x)) * (y / (5.0 + y))

        network.add_reaction(Reaction("synth_x", {}, {"x": 1}, k_x))
        network.add_reaction(Reaction("deg_x", {"x": 1}, {}, deg_x))
        network.add_reaction(Reaction("synth_y", {}, {"y": 1}, 1.0, propensity_fn=prop_synth_y))
        network.add_reaction(Reaction("deg_y", {"y": 1}, {}, deg_y))
        network.add_reaction(Reaction("synth_z", {}, {"z": 1}, 1.0, propensity_fn=prop_synth_z))
        network.add_reaction(Reaction("deg_z", {"z": 1}, {}, deg_z))

        initial_state = {"x": 10, "y": 0, "z": 0}
        return network, initial_state
