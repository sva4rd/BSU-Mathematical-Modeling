import math
import random
import numpy as np
import matplotlib.pyplot as plt


def bernoulli(n, p):
    return [int(random.random() < p) for _ in range(n)]


def binomial(m, p, n):
    return [sum(bernoulli(m, p)) for _ in range(n)]


def geometric(n, p):
    return [math.ceil(math.log(random.random()) / math.log(1 - p)) for _ in range(n)]


def poisson(n, lambda_):
    poisson_rvs = []
    for _ in range(n):
        L = math.exp(-lambda_)
        k = 0
        P = 1
        while P > L:
            k = k + 1
            P = P * random.random()
        poisson_rvs.append(k - 1)
    return poisson_rvs


# Количество реализаций
n = 1000

# Бернулли
p = 0.7
bernoulli_val = bernoulli(n, p)
bernoulli_mean = sum(bernoulli_val) / n
bernoulli_var = sum((x - bernoulli_mean) ** 2 for x in bernoulli_val) / (n - 1)
print(f'Бернулли: \n\tМат ожидание: {bernoulli_mean} (истинное значение {p}), \n\tдисперсия {bernoulli_var} '
      f'(истинное значение {p * (1 - p)})')

# Биномиальное
m = 5
p = 0.25
binomial_val = binomial(m, p, n)
binomial_mean = sum(binomial_val) / n
binomial_var = sum((x - binomial_mean) ** 2 for x in binomial_val) / (n - 1)
print(f'Биномиальное: \n\tМат ожидание: {binomial_mean} (истинное значение {m * p}), \n\tдисперсия {binomial_var} '
      f'(истинное значение {m * p * (1 - p)})')

# Геометрическое
p = 0.7
geometric_val = geometric(n, p)
geometric_mean = sum(geometric_val) / n
geometric_var = sum((x - geometric_mean) ** 2 for x in geometric_val) / (n - 1)
print(f'Геометрическое: \n\tМат ожидание: {geometric_mean} (истинное значение {1 / p}), \n\tдисперсия {geometric_var} '
      f'(истинное значение {(1 - p) / (p ** 2)})')

# Пуассона
lambda_ = 2
poisson_val = poisson(n, lambda_)
poisson_mean = sum(poisson_val) / n
poisson_var = sum((x - poisson_mean) ** 2 for x in poisson_val) / (n - 1)
print(f'Пуассона: \n\tМат ожидание: {poisson_mean} (истинное значение {lambda_}), \n\tдисперсия {poisson_var} '
      f'(истинное значение {lambda_})')

print("=" * 50)


# 1 Другой способ
def alternative_bernoulli(n, p):
    bernoulli_rvs = []
    for _ in range(n):
        u = random.random()
        if u < p:
            bernoulli_rvs.append(1)
        else:
            bernoulli_rvs.append(0)
    # bernoulli_rvs = np.random.binomial(1, p, n)
    return bernoulli_rvs


bernoulli_val = alternative_bernoulli(n, p)
bernoulli_mean = sum(bernoulli_val) / n
bernoulli_var = sum((x - bernoulli_mean) ** 2 for x in bernoulli_val) / (n - 1)
print(
    f'Альтернативное Бернулли: \n\tМат ожидание: {bernoulli_mean} (истинное значение {p}), \n\tдисперсия {bernoulli_var} '
    f'(истинное значение {p * (1 - p)})')


# 2 оценки коэффициентов эксцесса и асимметрии
def compute_kurtosis(data):
    data_len = len(data)
    mean = sum(data) / data_len
    variance = sum((x - mean) ** 2 for x in data) / data_len
    kurtosis = sum((x - mean) ** 4 for x in data) / (variance ** 2 * data_len) - 3
    return kurtosis


def compute_skewness(data):
    data_len = len(data)
    mean = sum(data) / data_len
    variance = sum((x - mean) ** 2 for x in data) / data_len
    skewness = sum((x - mean) ** 3 for x in data) / (variance ** (3 / 2) * data_len)
    return skewness


p = 0.7
bernoulli_samples = np.random.binomial(1, p, n)
bernoulli_skewness = compute_skewness(bernoulli_samples)
bernoulli_kurtosis = compute_kurtosis(bernoulli_samples)

m, p = 5, 0.25
binomial_samples = np.random.binomial(m, p, n)
binomial_skewness = compute_skewness(binomial_samples)
binomial_kurtosis = compute_kurtosis(binomial_samples)

p = 0.7
geometric_samples = np.random.geometric(p, n)
geometric_skewness = compute_skewness(geometric_samples)
geometric_kurtosis = compute_kurtosis(geometric_samples)

poisson_samples = np.random.poisson(lambda_, n)
poisson_skewness = compute_skewness(poisson_samples)
poisson_kurtosis = compute_kurtosis(poisson_samples)

print(f'\nБернулли: \n\tасимметрия {bernoulli_skewness}, истинная {(1 - 2 * p) / math.sqrt(p * (1 - p))}'
      f'\n\tэксцесс {bernoulli_kurtosis}, истинный {(1 - 2 * p) / (p * (1 - p))}')
print(f'Биномиальное: \n\tасимметрия {binomial_skewness}, истинная {(1 - 2 * p) / math.sqrt(m * p * (1 - p))}'
      f'\n\tэксцесс {binomial_kurtosis}, истинный {(1 - 6 * p * (1 - p)) / (m * p * (1 - p))}')
print(f'Геометрическое: \n\tасимметрия {geometric_skewness}, истинная {(2 - p) / math.sqrt(1 - p)}'
      f'\n\tэксцесс {geometric_kurtosis}, истинный {6 + p ** 2 / (1 - p)}')
print(f'Пуассона: \n\tасимметрия {poisson_skewness}, истинная {1 / math.sqrt(lambda_)}'
      f'\n\tэксцесс {poisson_kurtosis}, истинный {1 / lambda_}')

# Задаем параметры распределения
p = 0.3
sample_sizes = [100, 1000, 10000]  # Размеры выборок

# Создаем подложку для графиков
fig, axes = plt.subplots(len(sample_sizes), 1, figsize=(6, 8))

for i, sample_size in enumerate(sample_sizes):
    # Генерируем выборку из геометрического распределения
    sample = np.random.geometric(p, sample_size)

    # Вычисляем теоретические вероятности
    values = np.unique(sample)
    theoretical_probs = [(1-p)**(value-1) * p for value in values]

    # Вычисляем эмпирические вероятности
    empirical_probs = [np.mean(sample == value) for value in values]

    # Построение гистограммы
    width = 0.4  # Ширина столбцов
    axes[i].bar(values, theoretical_probs, width=width, alpha=0.5, label='Theoretical')
    axes[i].bar(values + width, empirical_probs, width=width, alpha=0.5, label='Empirical')
    # axes[i].set_xlabel('Values')
    axes[i].set_ylabel('Probabilities')
    axes[i].set_title(f'Histogram (n = {sample_size})')
    axes[i].legend()

plt.tight_layout()
plt.show()


# 4  график эмпирической функции распределения сравнить с графиком теоретической функции распределения
# Задаем параметры распределения
p = 0.3
sample_size = 1000  # Размер выборки

# Генерируем выборку из геометрического распределения
sample = np.random.geometric(p, sample_size)

# Вычисляем значения исходных данных
values = np.sort(sample)
n = len(values)
y = np.arange(1, n+1) / n

# Вычисляем теоретическую функцию распределения
x_theoretical = np.arange(1, np.max(values)+1)
y_theoretical = 1 - (1-p)**x_theoretical

# Построение графиков
plt.plot(values, y, label='Empirical')
plt.plot(x_theoretical, y_theoretical, label='Theoretical')
plt.xlabel('Values')
plt.ylabel('Cumulative Probability')
plt.title('Empirical vs Theoretical CDF')
plt.legend()
plt.show()
