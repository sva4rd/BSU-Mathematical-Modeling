import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# var9
M = 2147483648
n = 1000
a1 = 26094681
c1 = 196808461
a2 = 109243693
c2 = 100464893
K = 160


def sensor_generator(a0, c):
    b = max(c, M - c)
    while True:
        a0 = (b * a0) % M
        yield a0 / M


def print_results(brv_values, method_name):
    print(f"{method_name}:\n\ta1:\t\t{round(brv_values[0], 6)}\n\ta15:"
          f"\t{round(brv_values[14], 6)}\n\ta1000:\t{round(brv_values[999], 6)}")
    plt.hist(brv_values, bins=10)
    plt.title(method_name)
    plt.show()


def Multiplicative_congruent_method():
    D1 = sensor_generator(a1, c1)
    brv_list = [next(D1) for _ in range(n)]
    print_results(brv_list, "Multiplicative congruent method")
    return brv_list


def McLaren_Marsaglia_method():
    D1 = sensor_generator(a1, c1)
    D2 = sensor_generator(a2, c2)
    V = [next(D1) for _ in range(K)]
    c = [next(D2) for _ in range(n)]
    brv_list = []
    for i in range(n):
        s = int(c[i] * K)
        brv_list.append(V[s])
        V[s] = next(D1)
    print_results(brv_list, "McLaren-Marsaglia method")


a_values = Multiplicative_congruent_method()
McLaren_Marsaglia_method()

print("=" * 50)


def find_period(sequence):
    seen = {}
    for i, value in enumerate(sequence):
        if value in seen:
            return i - seen[value]
        seen[value] = i
    return -1


def compute_correlation_coefficients(data, t=30):
    coefficients = []
    for lag in range(1, t + 1):
        coeff = np.corrcoef(data[:-lag], data[lag:])[0, 1]
        coefficients.append(coeff)
    return coefficients


def covariance_test(data, t=30, e=0.05):
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)

    failed_lags = []
    failed_lags_val = []
    for lag in range(1, t + 1):
        cov = np.mean((data[lag:] - mean) * (data[:-lag] - mean))
        z = cov / np.sqrt(var ** 2 / (n - 1))

        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        if p_value < e:
            failed_lags.append(lag)
            failed_lags_val.append(p_value)

    return failed_lags, failed_lags_val


# 3
coefficients = compute_correlation_coefficients(a_values)
print(f"Коэффициенты корреляции:\n")
for i, coefficient in enumerate(coefficients, start=1):
    print(f"{i}: {coefficient}")

print(f"Интерпретация значений коэффициентов корреляции:"
      f"\nЕсли rτ близко к 1, это указывает на сильную прямую связь между at и at+τ. "
      f"То есть, когда at увеличивается, at+τ также увеличивается.")
print(f"Если rτ близко к -1, это указывает на сильную обратную связь между at и at+τ. "
      f"То есть, когда at увеличивается, at+τ уменьшается.")
print(f"Если rτ близко к 0, это указывает на отсутствие линейной связи между at и at+τ.")

# 2
data = np.random.normal(size=1000)
failed_lags, failed_lags_val = covariance_test(data)
print(f"\nТест ковариация. Тест не проходит для следующих значений лага: {failed_lags} {failed_lags_val}")

# 7
print(f"\nДлина периода\nВ генераторе МКМ длина периода может быть максимум равна M, "
      f"\nгде M - модуль, используемый в генераторе.\nОднако длина периода может быть меньше "
      f"в зависимости от выбранных параметров a и c.")
period_length = find_period(a_values)
print(f"Практическая длина периода для генератора МКМ: {period_length}")
