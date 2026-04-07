import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")


class TemperedStableProcess:
    """
    Класс для моделирования процесса Tempered Stable (эквивалент CGMY).
    Параметры (c, alpha, lambda_plus, lambda_minus) соответствуют (C, Y, M, G).
    """

    def __init__(
        self, c, alpha, lambda_plus, lambda_minus, T=1.0, n_steps=252, n_paths=1
    ):
        self.c = c
        self.alpha = alpha  # Индекс стабильности (аналог Y, < 2)
        self.lam_p = lambda_plus  # Затухание правого хвоста (аналог M)
        self.lam_m = lambda_minus  # Затухание левого хвоста (аналог G)
        self.T = T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.dt = T / n_steps

    def levy_density(self, x):
        """Плотность меры Леви для Tempered Stable процесса"""
        x = np.asarray(x)
        density = np.zeros_like(x, dtype=float)
        pos = x > 0
        neg = x < 0
        density[pos] = (
            self.c * np.exp(-self.lam_p * x[pos]) / (x[pos] ** (1 + self.alpha))
        )
        density[neg] = (
            self.c
            * np.exp(-self.lam_m * np.abs(x[neg]))
            / (np.abs(x[neg]) ** (1 + self.alpha))
        )
        return density

    def _small_jumps_moments(self, epsilon):
        """Вычисление моментов для диффузионной аппроксимации малых скачков"""
        # Аналитическая аппроксимация дисперсии малых скачков (возле нуля e^{-lam*x} ~ 1)
        var_approx = self.c * (epsilon ** (2 - self.alpha)) / (2 - self.alpha) * 2

        # Интегрирование для смещения (mean)
        def integrand_mean(x):
            return x * self.levy_density(x)

        try:
            mean_pos, _ = quad(integrand_mean, 0, epsilon)
            mean_neg, _ = quad(integrand_mean, -epsilon, 0)
            total_mean = mean_pos + mean_neg
        except:
            total_mean = (
                0.0  # Если интеграл расходится, считаем процесс симметричным в нуле
            )

        return total_mean, var_approx

    def _get_large_jump_sizes(self, lam, sign, n_jumps, epsilon):
        """Точная генерация больших скачков методом обратного преобразования CDF"""
        if n_jumps == 0:
            return np.array([])

        # Создаем сетку для хвоста распределения
        x_grid = np.linspace(epsilon, epsilon + 15 / lam, 2000)
        pdf = (1.0 / x_grid ** (1 + self.alpha)) * np.exp(-lam * x_grid)
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]  # Нормировка

        u = np.random.uniform(0, 1, n_jumps)
        jumps = np.interp(u, cdf, x_grid)
        return jumps * sign

    def simulate_paths(self, epsilon=0.05):
        """Симуляция траекторий методом Росински (разделение на малые и большие скачки)"""
        # 1. Аппроксимация малых скачков броуновским движением
        mu_small, var_small = self._small_jumps_moments(epsilon)

        paths = np.zeros((self.n_paths, self.n_steps + 1))
        time_grid = np.linspace(0, self.T, self.n_steps + 1)

        for i in range(self.n_paths):
            # Диффузионная часть
            if var_small > 0:
                dW = np.random.normal(
                    mu_small * self.dt, np.sqrt(var_small * self.dt), self.n_steps
                )
                cum_small = np.cumsum(dW)
            else:
                cum_small = np.zeros(self.n_steps)

            # 2. Моделирование больших скачков (Пуассоновский процесс)
            int_pos, _ = quad(lambda x: self.levy_density(x), epsilon, np.inf)
            int_neg, _ = quad(lambda x: self.levy_density(x), -np.inf, -epsilon)

            n_pos = np.random.poisson(int_pos * self.T)
            n_neg = np.random.poisson(int_neg * self.T)

            times_pos = np.random.uniform(0, self.T, n_pos) if n_pos > 0 else []
            times_neg = np.random.uniform(0, self.T, n_neg) if n_neg > 0 else []

            jumps_pos = self._get_large_jump_sizes(self.lam_p, 1, n_pos, epsilon)
            jumps_neg = self._get_large_jump_sizes(self.lam_m, -1, n_neg, epsilon)

            # Агрегация скачков по времени
            all_times = np.concatenate([times_pos, times_neg])
            all_jumps = np.concatenate([jumps_pos, jumps_neg])

            cum_jumps = np.zeros(self.n_steps)
            if len(all_times) > 0:
                # Находим индексы интервалов, в которые попали скачки
                indices = np.searchsorted(time_grid[1:], all_times)
                for idx, jump in zip(indices, all_jumps):
                    if idx < self.n_steps:
                        cum_jumps[idx] += jump

            cum_jumps = np.cumsum(cum_jumps)

            # Итоговая траектория
            paths[i, 1:] = cum_small + cum_jumps

        self.time_grid = time_grid
        self.paths = paths
        return paths

    def plot_analysis(self, scenario_name):
        """Визуализация траекторий и конечного распределения"""
        fig = plt.figure(figsize=(14, 5))
        fig.suptitle(scenario_name, fontsize=14, fontweight="bold")

        # График траекторий
        ax1 = fig.add_subplot(1, 2, 1)
        for i in range(min(15, self.n_paths)):  # Рисуем 15 траекторий
            ax1.plot(self.time_grid, self.paths[i], lw=1.5, alpha=0.8)
        ax1.set_title(f"Траектории процесса Tempered Stable")
        ax1.set_xlabel("Время (t)")
        ax1.set_ylabel("X(t)")
        ax1.grid(True, alpha=0.4)

        # Гистограмма конечных значений
        ax2 = fig.add_subplot(1, 2, 2)
        final_vals = self.paths[:, -1]
        ax2.hist(
            final_vals,
            bins=40,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )
        ax2.set_title(f"Распределение X(T) при T={self.T}")
        ax2.set_xlabel("Значение X(T)")
        ax2.set_ylabel("Плотность")
        ax2.grid(True, alpha=0.4)

        plt.tight_layout()
        plt.show()


# ==========================================
# ОСНОВНОЙ БЛОК: 4 Сценария
# ==========================================
if __name__ == "__main__":
    # Сценарии параметров: (c, alpha, lam_p, lam_m)
    scenarios = [
        {
            "name": "Сценарий 1: Симметричный, конечная вариация (alpha=0.6)",
            "params": (1.0, 0.6, 4.0, 4.0),
        },
        {
            "name": "Сценарий 2: Симметричный, бесконечная вариация (alpha=1.4)",
            "params": (0.5, 1.4, 5.0, 5.0),
        },
        {
            "name": "Сценарий 3: Асимметричный, правый скос (lam_p=2, lam_m=8)",
            "params": (1.0, 0.8, 2.0, 8.0),
        },
        {
            "name": "Сценарий 4: Высокая активность (почти Броуновское движение, alpha=1.8)",
            "params": (0.1, 1.8, 10.0, 10.0),
        },
    ]

    for i, sc in enumerate(scenarios, 1):
        print(f"\n{sc['name']}")
        print("-" * 50)

        # Создание и симуляция
        ts_process = TemperedStableProcess(
            *sc["params"], T=1.0, n_steps=252, n_paths=1000
        )
        paths = ts_process.simulate_paths(epsilon=0.05)

        # Статистика
        final_values = paths[:, -1]
        print(f"Среднее:            {np.mean(final_values):.4f}")
        print(f"Станд. отклонение:  {np.std(final_values):.4f}")
        print(f"Асимметрия (Skew):  {stats.skew(final_values):.4f}")
        print(f"Эксцесс (Kurtosis): {stats.kurtosis(final_values):.4f}")

        # Отрисовка
        ts_process.plot_analysis(sc["name"])
