import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ==========================================
# 1. Задание параметров (согласно отчету)
# ==========================================
T = 1.0  # Временной интервал [0, T]
N = 1000  # Количество шагов дискретизации (выбрано для плавности графиков)
dt = T / N  # Шаг дискретизации (Алгоритм, п.1)
M = 2000  # Количество траекторий для сбора статистики (Раздел 5)
plot_count = 20  # Количество траекторий для вывода на график

# ==========================================
# 2. Моделирование (Алгоритм, п. 2-5)
# ==========================================
# Вводится равномерная сетка времени
t = np.linspace(0, T, N + 1)

# Генерируется массив независимых случайных величин Z ~ N(0, 1) (M траекторий по N шагов)
Z = np.random.normal(0, 1, size=(M, N))

# Вычисляются приращения dW = sqrt(dt) * Z
dW = np.sqrt(dt) * Z

# Значение процесса вычисляется рекурсивно (через кумулятивную сумму)
# W_0 = 0
W = np.zeros((M, N + 1))
# W_{t_i} = W_{t_{i-1}} + dW_i
W[:, 1:] = np.cumsum(dW, axis=1)

# ==========================================
# 3. Визуализация результатов (Раздел 5)
# ==========================================
# Настройка полотна для 3-х графиков
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# --- График 1: Траектории процесса ---
for i in range(plot_count):
    axes[0].plot(t, W[i, :], lw=1)

axes[0].set_title(
    f"Моделирование {plot_count} траекторий\nВинеровского процесса", fontsize=10
)
axes[0].set_xlabel("Время (t)", fontsize=9)
axes[0].set_ylabel("W(t)", fontsize=9)
axes[0].grid(True, linestyle="--", alpha=0.6)

# --- График 2: Гистограмма распределения при t = T ---
W_T = W[:, -1]  # Срез всех траекторий в последний момент времени T

# Эмпирическая гистограмма
axes[1].hist(
    W_T,
    bins=40,
    density=True,
    alpha=0.5,
    color="skyblue",
    edgecolor="black",
    label="Эмпирическое",
)

# Теоретическая плотность N(0, T)
x_val = np.linspace(-3, 3, 200)
y_val = stats.norm.pdf(x_val, loc=0, scale=np.sqrt(T))
axes[1].plot(x_val, y_val, "r-", lw=2, label=f"Теоретическое $\mathcal{{N}}(0, {T})$")

axes[1].set_title(f"Распределение W(T) при T={T}", fontsize=10)
axes[1].set_xlabel("Значение W(T)", fontsize=9)
axes[1].set_ylabel("Плотность вероятности", fontsize=9)
axes[1].legend(fontsize=9)
axes[1].grid(True, linestyle="--", alpha=0.6)

# --- График 3: Дисперсия процесса ---
# Эмпирическая дисперсия (выборочная дисперсия по всем M траекториям для каждого шага t)
emp_var = np.var(W, axis=0)

axes[2].plot(t, emp_var, "b--", lw=2, label="Эмпирическая дисперсия")
# Теоретическая дисперсия Var(W_t) = t
axes[2].plot(t, t, "r-", lw=1.5, label="Теоретическая (Var=t)")

axes[2].set_title("Дисперсия процесса в зависимости от времени", fontsize=10)
axes[2].set_xlabel("Время (t)", fontsize=9)
axes[2].set_ylabel("Дисперсия", fontsize=9)
axes[2].legend(fontsize=9)
axes[2].grid(True, linestyle="--", alpha=0.6)

# Общая настройка и вывод
plt.tight_layout()
plt.show()
