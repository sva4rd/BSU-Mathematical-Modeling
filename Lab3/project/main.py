import random
import math
import numpy as np
from scipy.stats import chi2, laplace
import matplotlib.pyplot as plt


def calculate_mean(numbers):
    return sum(numbers) / len(numbers)


def calculate_variance(numbers, mean):
    return sum((x - mean) ** 2 for x in numbers) / (len(numbers) - 1)


# Задание 1
def simulate_normal_distribution(n, m, s, N=48):
    X = np.random.uniform(size=(n, N))
    X = X.sum(axis=1) - N / 2
    X = m + s * np.sqrt(12 / N) * X
    return X


# Задание 2
def simulate_chi_square_distribution(n, m):
    chi_square_values = []
    for _ in range(n):
        chi_square = sum((random.gauss(0, 1)) ** 2 for _ in range(m))
        chi_square_values.append(chi_square)
    return chi_square_values


def simulate_laplace_distribution(n, a):
    laplace_values = []
    for _ in range(n):
        u = random.uniform(-0.5, 0.5)
        b = -a * math.copysign(1, u) * math.log(1 - 2 * abs(u))
        laplace_values.append(b)
    return laplace_values


n = 10000
print(f"========Задание №1========")
parameters = [(0, 9), (-3, 16)]
for m, ss in parameters:
    s = math.sqrt(ss)
    simulations = simulate_normal_distribution(n, m, s)
    mean = calculate_mean(simulations)
    variance = calculate_variance(simulations, mean)
    print(f"Параметры: m = {m}, s^2 = {s ** 2}")
    print(f"Несмещенные оценки:\n\tмат. ожидание = {mean}\n\tдисперсия = {variance}")
    print(
        f"Сравнение с истинными значениями:\n\tразница между мат. ожиданиями = {abs(m - mean)}"
        f"\n\tразница между дисперсиями = {abs(s ** 2 - variance)}\n")

print(f"========Задание №2========")
parameters2 = [(4, simulate_chi_square_distribution, "1) χ^2-распределение"),
               (2, simulate_laplace_distribution, "\n2) Распределение Лапласа")]
for m, simulate_distribution, name in parameters2:
    print(name)
    simulations = simulate_distribution(n, m)
    mean = calculate_mean(simulations)
    variance = calculate_variance(simulations, mean)
    print(f"Параметры: m = {m}")
    print(f"Несмещенные оценки:\n\tмат. ожидание = {mean}\n\tдисперсия = {variance}")
    if name == "1) χ^2-распределение":
        print(f"Истинные значения:\n\tмат. ожидание = {m}\n\tдисперсия = {2 * m}")
        print(
            f"Сравнение с истинными значениями:\n\tразница между мат. ожиданиями = {abs(m - mean)}"
            f"\n\tразница между дисперсиями = {abs(2 * m - variance)}")
    else:
        print(f"Истинные значения:\n\tмат. ожидание = {0}\n\tдисперсия = {2 * m ** 2}")
        print(
            f"Сравнение с истинными значениями:\n\tразница между мат. ожиданиями = {abs(0 - mean)}"
            f"\n\tразница между дисперсиями = {abs(2 * m ** 2 - variance)}")

print("\n" + "=" * 50)


def simulate_mixture_distribution(n, m, a, pi):
    mixture_values = []
    for _ in range(n):
        if random.random() < pi:
            # Генерация случайной величины из χ²-распределения
            # value = np.random.chisquare(m)
            value = sum((random.gauss(0, 1)) ** 2 for _ in range(m))
        else:
            # Генерация случайной величины из распределения Лапласа
            u = random.uniform(-0.5, 0.5)
            value = -a * math.copysign(1, u) * math.log(1 - 2 * abs(u))
        mixture_values.append(value)
    return mixture_values


def compute_mixture_statistics(samples, pi):
    n = len(samples)
    mean_X = np.mean(samples)
    var_X = np.var(samples, ddof=1)

    if pi == 0.5:
        mixture_mean = 0.5 * mean_X + 0.5 * 0
        mixture_var = 0.5 * var_X + 0.5 * (mean_X ** 2)
    else:
        # Если pi != 0.5, необходимо вычислить математическое ожидание и дисперсию распределения Y
        # и затем использовать полные формулы для вычисления оценок
        pass

    return mixture_mean, mixture_var


def calculate_mixture_mean(m1, m2, pi):
    return m1 * pi + m2 * (1 - pi)


def calculate_mixture_variance(m1, m2, v1, v2, pi):
    return pi * (v1 + (m1 - calculate_mixture_mean(m1, m2, pi)) ** 2) + (1 - pi) * \
           (v2 + (m2 - calculate_mixture_mean(m1, m2, pi)) ** 2)


# 1 + 2 Смесь двух распределений с вычислением мат ожиданий и дисперсий
n = 100000
m = 4
a = 2
pi = 0.5

mixture_simulations = simulate_mixture_distribution(n, m, a, pi)
mean = calculate_mean(mixture_simulations)
variance = calculate_variance(mixture_simulations, mean)

true_mean = m * pi + 0 * (1 - pi)  # сумма мат ожиданий с весовыми кэфами
true_variance = (2 * m + m * m) * pi + (2 * a ** 2 + 0 * 0) * (1 - pi) - true_mean ** 2
# сумма((дисперсия + мат ожид ^ 2)*pi) - мат ожид смеси ^ 2

print(f"Смесь двух распределений")
print(f"Несмещенные оценки:\n\tмат. ожидание = {mean}\n\tдисперсия = {variance}")
print(f"Истинные значения:\n\tмат. ожидание = {true_mean}\n\tдисперсия = {true_variance}")

# 3 бокс-мюллер
n = 10000

# Генерация равномерно распределенных случайных величин на интервале (0, 1)
u1 = np.random.uniform(0, 1, n)
u2 = np.random.uniform(0, 1, n)

# Преобразование Бокса — Мюллера
z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

mean_estimate = np.mean(z1)
variance_estimate = np.var(z1, ddof=1)
true_mean = 0
true_variance = 1
print(f"\nПреобразование Бокса-Мюллера")
print(f"Несмещенные оценки:\n\tмат. ожидание = {mean_estimate}\n\tдисперсия = {variance_estimate}")
print(f"Истинные значения:\n\tмат. ожидание = {true_mean}\n\tдисперсия = {true_variance}")


# 4 оценить коэффициент корреляции
def correlation_coefficient(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    correlation = numerator / denominator
    return correlation


# Выбор элементов на четных и нечетных позициях
even_positions = z1[::2]
odd_positions = z1[1::2]

# Оценка коэффициента корреляции
correlation_coefficient = correlation_coefficient(even_positions, odd_positions)
# correlation_coefficient = np.corrcoef(even_positions, odd_positions)[0, 1]

print("\nОценка коэффициента корреляции:")
print(f"Коэффициент корреляции: {correlation_coefficient}")


# 5 гистограммы
n = 10000
m = 4
a = 2

chi_square_values = simulate_chi_square_distribution(n, m)
laplace_values = simulate_laplace_distribution(n, a)

# Построение гистограммы для распределения хи-квадрат
plt.figure()
plt.hist(chi_square_values, bins=30, density=True, label='Simulated Data')
x = np.linspace(0, max(chi_square_values), 100)
plt.plot(x, chi2.pdf(x, m), 'r-', lw=2, label='Theoretical PDF')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Chi-Square Distribution')
plt.legend()

# Построение гистограммы для распределения Лапласа
plt.figure()
plt.hist(laplace_values, bins=30, density=True, label='Simulated Data')
x = np.linspace(min(laplace_values), max(laplace_values), 100)
plt.plot(x, laplace.pdf(x, scale=a), 'r-', lw=2, label='Theoretical PDF')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Laplace Distribution')
plt.legend()

plt.show()





