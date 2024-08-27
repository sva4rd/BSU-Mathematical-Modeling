import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt


def get_next_markov(currentI, P):
    rand = random.random()
    i = 0
    k = 0
    while i < rand:
        i += P[currentI][k]
        k += 1
    return k - 1


def count_hi(l, A, f, h_vector, pi, p):
    rand = random.random()
    i = 0
    k = 0
    while i < rand:
        i += pi[k]
        k += 1
    current = k - 1
    if h_vector[current] == 0:
        return 0
    Q = h_vector[current] / pi[current]
    hi = Q * f[current]
    for i in range(l):
        next_markov = get_next_markov(current, p)
        Q *= A[current][next_markov] / p[current][next_markov]
        hi += Q * f[next_markov]
        current = next_markov
    return hi


def monte_carlo_solving(markov_len, markov_number, A, f):
    size = len(A)
    p = [[1 / size] * size for _ in range(size)]
    pi = [1 / size] * size
    x = np.zeros(size)
    h = np.identity(size)

    for j in range(size):
        x[j] = sum(count_hi(markov_len, A, f, h[:, j], pi, p) for _ in range(markov_number)) / markov_len

    return x


def norm(a, b):
    return np.linalg.norm(a - b)


def get_numbers():
    return (2 ** x for x in range(13))


def calculate_iteration_number(func, rez, A, f):
    return [norm(func(1000, i, A, f), rez) for i in get_numbers()]


def calculate_markov_len(func, rez, A, f):
    return [norm(func(i, 1000, A, f), rez) for i in get_numbers()]


def calculate_iteration_number_and__markov_len(func, rez, A, f):
    return [norm(func(i, i, A, f), rez) for i in get_numbers()]


def main():
    A = [[1.0, -0.4, -0.1],
         [0.4, 0.7, -0.1],
         [0.3, 0.2, 1.0]]
    f = [-1, 5, 4]
    B = np.eye(3) - A
    rez = np.linalg.solve(A, f)
    print("Точное решение СЛАУ:", rez)
    print("Решение СЛАУ методом Монте-Карло:", monte_carlo_solving(1000, 1000, B.copy(), f))

    offset_both = calculate_iteration_number_and__markov_len(monte_carlo_solving, rez, B.copy(), f)
    draw(offset_both, "Length & Count")


def draw(real, str):
    matplotlib.rc('ytick', labelsize=15)
    matplotlib.rc('xtick', labelsize=15)
    plt.figure(figsize=(10, 8))
    x = list(get_numbers())
    plt.plot(x, [0] * len(x))
    plt.plot(x, real)
    plt.xscale('log')
    plt.xticks(x, x)
    plt.xlabel(str, fontsize=12)
    plt.ylabel('Norm', fontsize=12)
    plt.show()


if __name__ == '__main__':
    main()
