import argparse
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    N: int = 1000
    p: float = 0.8
    iterations: int = 1000
    p_sim_int: int = 50
    p_vals_int: int = 100

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "SimulationConfig":
        return cls(**vars(args))


def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(
        description="Statistical Simulation Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-N", type=int, default=1000, help="Sample size N")
    parser.add_argument("-p", type=float, default=0.8, help="Probability p")
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Number of iterations"
    )
    parser.add_argument(
        "--p-sim",
        type=int,
        default=50,
        dest="p_sim_int",
        help="Points for p simulation",
    )
    parser.add_argument(
        "--p-vals",
        type=int,
        default=100,
        dest="p_vals_int",
        help="Points for p theoretical values",
    )

    return SimulationConfig.from_namespace(parser.parse_args())
