from typing import Any

import numpy as np
from scipy import stats

GENERATOR = np.random.default_rng(seed=42)

def task_1(
        N: int,
        p: float,
        k_n: np.ndarray[Any, np.dtype[np.int_]] | Any
    ) -> tuple[np.int_, float]:

    x: np.ndarray[Any, np.dtype[np.int_]] = np.where(
        GENERATOR.random(size=N) <= p,
        k_n,
        1 - k_n
    )

    Y = np.sum(x)
    S_quad: float = Y * p + (N - Y) * (1 - p)

    return Y, S_quad

def task_2(N: int, p: float, Y: np.int_) -> np.intp:
    S_1: np.ndarray[Any, np.dtype[np.float16]] = stats.binom.pmf(
        k=np.arange(0, Y + 1),
        n=Y,
        p=p
    )
    S_2: np.ndarray[Any, np.dtype[np.float16]] = stats.binom.pmf(
        k=np.arange(0, N-Y + 1),
        n=N-Y,
        p=1-p
    )

    S = np.convolve(S_1, S_2, mode="full")
    cdf = np.cumsum(S)
    median_index = np.argmax(cdf >= 0.5)

    return median_index


N = 1000
p = 0.01

iterations = 1000

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

theoretical_S_original = N * 0.5
theoretical_error_1 = N * p * (1-p)
theoretical_error_2 = np.sqrt((2 * theoretical_error_1)/np.pi)

print(f"Теоретичне значення для початкової суми: {theoretical_S_original:.3f}")
print(f"Усереднений оригінал: {np.mean(S_originals):.3f}")
print()
print(f"Теоретичний баєсівський ризик: {theoretical_error_1:.3f}")
print(f"Середня квадратична помилка для оцінки: {np.mean(errors_1):.3f}")
print()
print(f"Теоретичний баєсівський ризик для модуля відхилення: {theoretical_error_2:.3f}")
print(f"Середній модуль відхилення для оцінки: {np.mean(errors_2):.3f}")
