from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm

from labs.lab1.parse import parse_args

GENERATOR = np.random.default_rng(seed=42)


def task_1(
    N: int, p: float, k_n: np.ndarray[Any, np.dtype[np.int_]] | Any
) -> tuple[np.int_, float]:

    x: np.ndarray[Any, np.dtype[np.int_]] = np.where(
        GENERATOR.random(size=N) <= p, k_n, 1 - k_n
    )

    Y = np.sum(x)
    S_quad: float = Y * p + (N - Y) * (1 - p)

    return Y, S_quad


def task_2(N: int, p: float, Y: np.int_) -> np.intp:
    S_1: np.ndarray[Any, np.dtype[np.float16]] = stats.binom.pmf(
        k=np.arange(0, Y + 1), n=Y, p=p
    )
    S_2: np.ndarray[Any, np.dtype[np.float16]] = stats.binom.pmf(
        k=np.arange(0, N - Y + 1), n=N - Y, p=1 - p
    )

    S = np.convolve(S_1, S_2, mode="full")
    cdf = np.cumsum(S)
    median_index = np.argmax(cdf >= 0.5)

    return median_index


def experiments(N: int, p: float, iterations: int) -> tuple[float, float, float]:

    S_originals = np.zeros(iterations)
    errors_1 = np.zeros(iterations)
    errors_2 = np.zeros(iterations)

    for iteration in range(iterations):
        k_n = GENERATOR.integers(0, 1 + 1, size=N)
        S_original = np.sum(k_n)
        S_originals[iteration] = S_original

        Y, S_quad = task_1(N, p, k_n)
        error_1 = (S_quad - S_original) ** 2
        errors_1[iteration] = error_1

        median_index = task_2(N, p, Y)
        error_2 = np.abs(median_index - S_original)
        errors_2[iteration] = error_2

    return np.mean(S_originals), np.mean(errors_1), np.mean(errors_2)


def results_for_single_p(N: int, p: float, iterations: int) -> None:

    S_originals_mean, errors_1_mean, errors_2_mean = experiments(N, p, iterations)

    theoretical_S_original = N * 0.5
    theoretical_error_1 = N * p * (1 - p)
    theoretical_error_2 = np.sqrt((2 * theoretical_error_1) / np.pi)

    print(f"N = {N} | p = {p}")
    print()
    print(f"Теоретичне значення для початкової суми: {theoretical_S_original:.3f}")
    print(f"Усереднений оригінал: {S_originals_mean:.3f}")
    print()
    print(f"Теоретичний баєсівський ризик: {theoretical_error_1:.3f}")
    print(f"Середня квадратична помилка для оцінки: {errors_1_mean:.3f}")
    print()
    print(
        f"Теоретичний баєсівський ризик абсолютної помилки: {theoretical_error_2:.3f}"
    )
    print(f"Середня абсолютна помилка для оцінки: {errors_2_mean:.3f}")
    print()


def visualize_errors_for_multiple_p(
    N: int, iterations: int, p_sim_int: int, p_vals_int: int
) -> None:

    p_sim = np.linspace(0, 1, p_sim_int)
    p_vals = np.linspace(0, 1, p_vals_int)
    reversed_needed: np.ndarray = np.array(
        [experiments(N, p, iterations) for p in tqdm(p_sim, desc="Simulating p-values")]
    )
    _, errors_1_mean, errors_2_mean = reversed_needed.T

    theo_mse = N * p_vals * (1 - p_vals)
    theo_mae = np.sqrt(2 * theo_mse / np.pi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(p_vals, theo_mse, "r-", label="Теорія")
    ax1.plot(p_sim, errors_1_mean, "b*", label="Симуляція")
    ax1.set(title="Квадратична помилка", xlabel="p", ylabel="MSE")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(p_vals, theo_mae, "r-", label="Теорія")
    ax2.plot(p_sim, errors_2_mean, "b*", label="Симуляція")
    ax2.set(title="Абсолютна помилка", xlabel="p", ylabel="MAE")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config = parse_args()

    results_for_single_p(N=config.N, p=config.p, iterations=config.iterations)
    visualize_errors_for_multiple_p(
        N=config.N,
        iterations=config.iterations,
        p_sim_int=config.p_sim_int,
        p_vals_int=config.p_vals_int,
    )
