"""Interactive CLI demo for BioKin Gillespie SSA gene circuit simulations."""
from __future__ import annotations
import argparse
import sys
import numpy as np

from biokin.circuits import Repressilator, GeneticToggleSwitch, FeedForwardLoop
from biokin.simulation import GillespieSSA, TauLeaping

def _ascii_plot(time: np.ndarray, values: np.ndarray, width: int = 60, height: int = 12, label: str = "") -> str:
    if len(time) < 2:
        return f"{label}: insufficient data"
    
    # Resample to width
    t_min, t_max = time[0], time[-1]
    v_min, v_max = float(np.min(values)), float(np.max(values))
    if v_max == v_min:
        v_max = v_min + 1.0
    
    grid = [[" " for _ in range(width)] for _ in range(height)]
    
    for i in range(len(time)):
        x = int((time[i] - t_min) / (t_max - t_min + 1e-12) * (width - 1))
        y = int((values[i] - v_min) / (v_max - v_min + 1e-12) * (height - 1))
        y = height - 1 - y
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        grid[y][x] = "*"
    
    lines = []
    header = f"{label} [{v_min:.1f} .. {v_max:.1f}] over t=[{t_min:.2f}, {t_max:.2f}]"
    lines.append(header)
    lines.append("+" + "-" * width + "+")
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append("+" + "-" * width + "+")
    return "\n".join(lines)


def demo_repressilator(t_max: float = 200.0, seed: int = 42) -> None:
    print("=" * 70)
    print("BioKin Demo: Elowitz-Leibler Repressilator (Genetic Oscillator)")
    print("=" * 70)
    
    network, initial = Repressilator.build_network()
    ssa = GillespieSSA(network, seed=seed)
    result = ssa.run(initial, t_max=t_max, max_steps=200_000)
    
    print(f"Simulation steps: {len(result.time)}")
    print(f"Final time: {result.time[-1]:.2f}")
    print(f"Final protein counts: TetR={result.trajectories['p_tetr'][-1]}, "
          f"cI={result.trajectories['p_ci'][-1]}, LacI={result.trajectories['p_laci'][-1]}")
    print()
    
    # Subsample for plot
    n = min(500, len(result.time))
    idx = np.linspace(0, len(result.time) - 1, n).astype(int)
    t_plot = result.time[idx]
    
    print(_ascii_plot(t_plot, result.trajectories["p_tetr"][idx], label="p_TetR"))
    print()
    print(_ascii_plot(t_plot, result.trajectories["p_ci"][idx], label="p_cI"))
    print()
    print(_ascii_plot(t_plot, result.trajectories["p_laci"][idx], label="p_LacI"))
    print()
    print("✓ Oscillatory dynamics expected under strong repression (n>1, alpha>>alpha0).")


def demo_toggle(t_max: float = 100.0, seed: int = 7) -> None:
    print("=" * 70)
    print("BioKin Demo: Gardner Genetic Toggle Switch (Bistability)")
    print("=" * 70)
    
    network, initial = GeneticToggleSwitch.build_network()
    ssa = GillespieSSA(network, seed=seed)
    result = ssa.run(initial, t_max=t_max)
    
    print(f"Simulation steps: {len(result.time)}")
    print(f"Final state: u={result.trajectories['u'][-1]}, v={result.trajectories['v'][-1]}")
    print()
    
    n = min(400, len(result.time))
    idx = np.linspace(0, len(result.time) - 1, n).astype(int)
    t_plot = result.time[idx]
    
    print(_ascii_plot(t_plot, result.trajectories["u"][idx], label="u (Repressor 1)"))
    print()
    print(_ascii_plot(t_plot, result.trajectories["v"][idx], label="v (Repressor 2)"))
    print()
    print("✓ Bistable switch: system locks into high-u or high-v state.")


def demo_ffl(t_max: float = 50.0, seed: int = 3) -> None:
    print("=" * 70)
    print("BioKin Demo: Coherent Type-1 Feed-Forward Loop (AND-gate)")
    print("=" * 70)
    
    network, initial = FeedForwardLoop.build_network()
    ssa = GillespieSSA(network, seed=seed)
    result = ssa.run(initial, t_max=t_max)
    
    print(f"Simulation steps: {len(result.time)}")
    print(f"Final: X={result.trajectories['x'][-1]}, Y={result.trajectories['y'][-1]}, Z={result.trajectories['z'][-1]}")
    print()
    
    n = min(300, len(result.time))
    idx = np.linspace(0, len(result.time) - 1, n).astype(int)
    t_plot = result.time[idx]
    
    print(_ascii_plot(t_plot, result.trajectories["x"][idx], label="X (Input)"))
    print()
    print(_ascii_plot(t_plot, result.trajectories["y"][idx], label="Y (Intermediate)"))
    print()
    print(_ascii_plot(t_plot, result.trajectories["z"][idx], label="Z (Output)"))
    print()
    print("✓ FFL AND-gate: Z activates only after both X and Y accumulate.")


def demo_tau_leap(t_max: float = 50.0, seed: int = 11) -> None:
    print("=" * 70)
    print("BioKin Demo: Adaptive Tau-Leaping vs Exact SSA (Simple Birth-Death)")
    print("=" * 70)
    
    from biokin.model import Reaction, ReactionNetwork
    
    net = ReactionNetwork()
    net.add_species("A")
    net.add_reaction(Reaction("birth", {}, {"A": 1}, 10.0))
    net.add_reaction(Reaction("death", {"A": 1}, {}, 0.1))
    
    initial = {"A": 50}
    
    ssa = GillespieSSA(net, seed=seed)
    res_ssa = ssa.run(initial, t_max=t_max)
    
    tau = TauLeaping(net, epsilon=0.03, seed=seed)
    res_tau = tau.run(initial, t_max=t_max, dt=0.05)
    
    print(f"SSA steps: {len(res_ssa.time)}, final A={res_ssa.trajectories['A'][-1]}")
    print(f"Tau-Leap steps: {len(res_tau.time)}, final A={res_tau.trajectories['A'][-1]}")
    print(f"Mean A (SSA): {np.mean(res_ssa.trajectories['A']):.2f}")
    print(f"Mean A (Tau): {np.mean(res_tau.trajectories['A']):.2f}")
    print()
    print("✓ Tau-leaping accelerates simulation while preserving approximate statistics.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m biokin.demo",
        description="BioKin: Gillespie SSA Gene Regulatory Network Demo",
    )
    parser.add_argument(
        "--circuit",
        choices=["repressilator", "toggle", "ffl", "tau", "all"],
        default="all",
        help="Which circuit to demonstrate",
    )
    parser.add_argument("--t-max", type=float, default=None, help="Simulation horizon")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args(argv)
    
    if args.circuit in ("repressilator", "all"):
        demo_repressilator(t_max=args.t_max or 200.0, seed=args.seed)
        print()
    if args.circuit in ("toggle", "all"):
        demo_toggle(t_max=args.t_max or 100.0, seed=args.seed)
        print()
    if args.circuit in ("ffl", "all"):
        demo_ffl(t_max=args.t_max or 50.0, seed=args.seed)
        print()
    if args.circuit in ("tau", "all"):
        demo_tau_leap(t_max=args.t_max or 50.0, seed=args.seed)
        print()
    
    print("BioKin demo complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
