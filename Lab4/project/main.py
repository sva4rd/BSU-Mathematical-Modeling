import numpy as np
import math
from scipy.integrate import quad, nquad
import matplotlib.pyplot as plt


def monte_carlo_integration(func, a, b, N):
    sum_func = 0
    x_values = np.linspace(a, b, N)
    for x in x_values:
        sum_func += func(x)
    integral = (b - a) * sum_func / N
    return integral


def monte_carlo_integration_2(func, a1, b1, a2, b2, N):
    sum_func = 0
    x_values = np.linspace(a1, b1, N)
    y_values = np.linspace(a2, b2, N)
    for i in range(N):
        sum_func += func(x_values[i], y_values[i])
    sum_func *= (b1 - a1) * (b2 - a2) / N
    return sum_func


def f1(x):
    return math.exp(-x) / (x * math.sqrt(1 + x ** 3))


def f2(x, y):
    return math.atan(x + y)


def main():
    a = 4
    b = 1000
    N = 1000000

    approximation = monte_carlo_integration(f1, a, b, N)
    print("Приближенное значение интеграла 1:", approximation)

    result, _ = quad(f1, a, np.inf)
    print("Приближенное значение интеграла 1 (мат. пакет):", result)

    a1 = math.exp(1)
    b1 = math.pi
    a2 = math.exp(3)
    b2 = math.pi ** 3

    approximation = monte_carlo_integration_2(f2, a1, b1, a2, b2, N)
    print("Приближенное значение интеграла 2 :", approximation)

    result, _ = nquad(f2, [[a1, b1], [a2, b2]])
    print("Приближенное значение интеграла 2 (мат. пакет):", result)

    showGraph()


def showGraph():
    a = 4
    b = 1000
    a1 = math.exp(1)
    b1 = math.pi
    a2 = math.exp(3)
    b2 = math.pi ** 3

    # Создание графика зависимости точности от числа итераций
    n_values = np.arange(1, 10000, 10)
    errors1 = []
    errors2 = []

    # Вычисление ошибок оценок интегралов для разных значений n
    for n in n_values:
        integral1 = monte_carlo_integration(f1, a, b, n)
        exact_integral1, _ = quad(f1, a, np.inf)
        errors1.append(abs(integral1 - exact_integral1))

        integral2 = monte_carlo_integration_2(f2, a1, b1, a2, b2, n)
        exact_integral2, _ = nquad(f2, [[a1, b1], [a2, b2]])
        errors2.append(abs(integral2 - exact_integral2))

    plt.figure(figsize=(8, 6))
    plt.plot(n_values, errors1, label='Величина ошибки интеграла 1')
    plt.plot(n_values, errors2, label='Величина ошибки интеграла 2')
    plt.ylim(-0.01, 0.15)
    plt.xlim(0, 1000)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Convergence of Monte Carlo Integration')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()

