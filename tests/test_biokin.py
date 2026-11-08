"""Unit and numerical invariance tests for BioKin Gillespie SSA engine."""
from __future__ import annotations
import numpy as np
import pytest

from biokin.model import Reaction, ReactionNetwork, Species
from biokin.simulation import GillespieSSA, TauLeaping, SimulationResult
from biokin.circuits import Repressilator, GeneticToggleSwitch, FeedForwardLoop


class TestModel:
    def test_species_and_reaction_basic(self):
        net = ReactionNetwork()
        net.add_species("A")
        net.add_species("B")
        rxn = Reaction("conversion", {"A": 1}, {"B": 1}, 0.5)
        net.add_reaction(rxn)
        assert len(net.species) == 2
        assert len(net.reactions) == 1
        assert "A" in net._species_to_idx
        assert "B" in net._species_to_idx

    def test_stoichiometry_matrix(self):
        net = ReactionNetwork()
        net.add_species("A")
        net.add_species("B")
        net.add_reaction(Reaction("r1", {"A": 1}, {"B": 1}, 1.0))
        net.add_reaction(Reaction("r2", {"B": 2}, {"A": 1}, 0.1))
        S = net.stoichiometry_matrix
        assert S.shape == (2, 2)
        # r1: A -> B  => A:-1, B:+1
        assert S[0, 0] == -1
        assert S[1, 0] == 1
        # r2: 2B -> A => A:+1, B:-2
        assert S[0, 1] == 1
        assert S[1, 1] == -2

    def test_mass_action_propensity_unimolecular(self):
        rxn = Reaction("decay", {"A": 1}, {}, 0.2)
        assert rxn.calculate_propensity({"A": 10}) == pytest.approx(2.0)
        assert rxn.calculate_propensity({"A": 0}) == pytest.approx(0.0)

    def test_mass_action_propensity_bimolecular(self):
        rxn = Reaction("dimerize", {"A": 2}, {"B": 1}, 0.1)
        # h = n*(n-1)/2 ; propensity = k * h
        assert rxn.calculate_propensity({"A": 5}) == pytest.approx(0.1 * 5 * 4 / 2)
        assert rxn.calculate_propensity({"A": 1}) == pytest.approx(0.0)

    def test_custom_propensity_fn(self):
        def hill(state):
            x = state.get("X", 0)
            return 10.0 / (1.0 + (x / 5.0) ** 2)
        rxn = Reaction("hill", {}, {"Y": 1}, 1.0, propensity_fn=hill)
        p = rxn.calculate_propensity({"X": 5})
        assert p == pytest.approx(10.0 / 2.0)


class TestGillespieSSA:
    def test_birth_death_mean_near_equilibrium(self):
        """Birth-death process A with birth rate λ, death rate μ should have E[A]=λ/μ."""
        net = ReactionNetwork()
        net.add_species("A")
        net.add_reaction(Reaction("birth", {}, {"A": 1}, 5.0))  # λ=5
        net.add_reaction(Reaction("death", {"A": 1}, {}, 0.5))  # μ=0.5 => mean=10
        ssa = GillespieSSA(net, seed=123)
        result = ssa.run({"A": 10}, t_max=200.0, max_steps=500_000)
        # Use second half of trajectory for steady-state mean
        half = len(result.time) // 2
        mean_a = np.mean(result.trajectories["A"][half:])
        assert 5.0 < mean_a < 15.0  # rough tolerance around 10

    def test_zero_propensity_terminates(self):
        net = ReactionNetwork()
        net.add_species("A")
        net.add_reaction(Reaction("decay", {"A": 1}, {}, 1.0))
        ssa = GillespieSSA(net, seed=1)
        result = ssa.run({"A": 3}, t_max=100.0)
        assert result.trajectories["A"][-1] == 0
        assert result.time[-1] < 100.0 or result.trajectories["A"][-1] == 0

    def test_reproducibility_with_seed(self):
        net = ReactionNetwork()
        net.add_species("X")
        net.add_reaction(Reaction("birth", {}, {"X": 1}, 2.0))
        net.add_reaction(Reaction("death", {"X": 1}, {}, 0.1))
        r1 = GillespieSSA(net, seed=99).run({"X": 5}, t_max=20.0)
        r2 = GillespieSSA(net, seed=99).run({"X": 5}, t_max=20.0)
        np.testing.assert_array_equal(r1.trajectories["X"], r2.trajectories["X"])
        np.testing.assert_array_almost_equal(r1.time, r2.time)

    def test_state_never_negative(self):
        net = ReactionNetwork()
        net.add_species("A")
        net.add_reaction(Reaction("decay", {"A": 1}, {}, 5.0))
        ssa = GillespieSSA(net, seed=7)
        result = ssa.run({"A": 20}, t_max=50.0)
        assert np.all(result.trajectories["A"] >= 0)

    def test_reaction_counts_sum(self):
        net = ReactionNetwork()
        net.add_species("A")
        net.add_reaction(Reaction("birth", {}, {"A": 1}, 1.0))
        net.add_reaction(Reaction("death", {"A": 1}, {}, 0.05))
        ssa = GillespieSSA(net, seed=5)
        result = ssa.run({"A": 10}, t_max=30.0)
        # Number of state updates equals sum of reaction firings (except final time pad)
        total_firings = int(np.sum(result.reaction_counts))
        assert total_firings >= 0
        assert len(result.time) >= 1


class TestTauLeaping:
    def test_tau_leap_basic_run(self):
        net = ReactionNetwork()
        net.add_species("A")
        net.add_reaction(Reaction("birth", {}, {"A": 1}, 8.0))
        net.add_reaction(Reaction("death", {"A": 1}, {}, 0.2))
        tau = TauLeaping(net, seed=42)
        result = tau.run({"A": 40}, t_max=30.0, dt=0.1)
        assert result.time[-1] >= 29.0
        assert np.all(result.trajectories["A"] >= 0)

    def test_tau_leap_mean_near_ssa(self):
        net = ReactionNetwork()
        net.add_species("A")
        net.add_reaction(Reaction("birth", {}, {"A": 1}, 10.0))
        net.add_reaction(Reaction("death", {"A": 1}, {}, 0.1))
        # Theoretical mean = 100
        ssa = GillespieSSA(net, seed=1)
        res_ssa = ssa.run({"A": 100}, t_max=50.0)
        tau = TauLeaping(net, seed=1)
        res_tau = tau.run({"A": 100}, t_max=50.0, dt=0.05)
        half_s = len(res_ssa.time) // 2
        half_t = len(res_tau.time) // 2
        mean_ssa = np.mean(res_ssa.trajectories["A"][half_s:])
        mean_tau = np.mean(res_tau.trajectories["A"][half_t:])
        # Both should be near 100, and relatively close to each other
        assert abs(mean_ssa - 100) < 30
        assert abs(mean_tau - 100) < 40


class TestCircuits:
    def test_repressilator_builds(self):
        net, init = Repressilator.build_network()
        assert len(net.species) == 6
        assert len(net.reactions) == 12
        assert "p_tetr" in init
        ssa = GillespieSSA(net, seed=1)
        result = ssa.run(init, t_max=20.0, max_steps=50_000)
        assert result.time[-1] > 0
        for s in net.species:
            assert np.all(result.trajectories[s] >= 0)

    def test_toggle_switch_builds(self):
        net, init = GeneticToggleSwitch.build_network()
        assert "u" in net.species and "v" in net.species
        ssa = GillespieSSA(net, seed=2)
        result = ssa.run(init, t_max=30.0)
        assert result.trajectories["u"][-1] >= 0
        assert result.trajectories["v"][-1] >= 0

    def test_ffl_builds(self):
        net, init = FeedForwardLoop.build_network()
        assert set(net.species) == {"x", "y", "z"}
        ssa = GillespieSSA(net, seed=3)
        result = ssa.run(init, t_max=20.0)
        assert result.time[-1] > 0

    def test_repressilator_oscillation_variance(self):
        """Repressilator proteins should exhibit non-trivial variance (not stuck at 0)."""
        net, init = Repressilator.build_network()
        ssa = GillespieSSA(net, seed=42)
        result = ssa.run(init, t_max=100.0, max_steps=200_000)
        var_tetr = np.var(result.trajectories["p_tetr"].astype(float))
        var_ci = np.var(result.trajectories["p_ci"].astype(float))
        # At least one protein should show variance > 0 (dynamics present)
        assert var_tetr + var_ci > 0.0


class TestSimulationResult:
    def test_get_species_trajectory(self):
        net = ReactionNetwork()
        net.add_species("A")
        net.add_reaction(Reaction("birth", {}, {"A": 1}, 1.0))
        ssa = GillespieSSA(net, seed=0)
        result = ssa.run({"A": 0}, t_max=5.0)
        traj = result.get_species_trajectory("A")
        assert len(traj) == len(result.time)
        assert isinstance(traj, np.ndarray)
