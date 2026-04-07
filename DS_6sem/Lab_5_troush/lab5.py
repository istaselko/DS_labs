import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ==========================================
# 1. Теоретический алгоритм генерации (Формулы 2 и 3)
# ==========================================
def generate_stable(alpha, beta, size):
    """
    Генерация независимых копий стандартной альфа-устойчивой величины 
    согласно алгоритму из Теоремы 2.
    """
    # Особый случай для броуновского движения (alpha = 2)
    # При alpha=2 распределение вырождается в нормальное N(0, 2)
    if alpha == 2.0:
        return np.random.normal(0, np.sqrt(2), size)
    
    # 1. Генерируются две независимые случайные величины V и W
    V = np.random.uniform(-np.pi/2, np.pi/2, size)
    W = np.random.exponential(1, size)
    
    # 2. Вычисляются константы C_{alpha, beta} и D_{alpha, beta}
    term = beta * np.tan(np.pi * alpha / 2)
    C_alpha_beta = np.arctan(term) / alpha
    D_alpha_beta = (np.cos(np.arctan(term))) ** (-1 / alpha)
    
    # 3. Вычисляется значение случайной величины X
    part1 = np.sin(alpha * (V + C_alpha_beta)) / (np.cos(V) ** (1 / alpha))
    part2 = (np.cos(V - alpha * (V + C_alpha_beta)) / W) ** ((1 - alpha) / alpha)
    
    X = D_alpha_beta * part1 * part2
    return X

# ==========================================
# 2. Моделирование процесса Леви (Формула 4)
# ==========================================
def simulate_levy_process(alpha, beta, M=2000, N=1000, T=1.0):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    
    # Генерация матрицы приращений
    X = generate_stable(alpha, beta, size=(M, N))
    
    # Масштабирование приращений: tau^(1/alpha) * X_i
    scale_factor = dt ** (1 / alpha)
    dW = scale_factor * X
    
    # Итерационное вычисление траекторий через кумулятивную сумму
    L = np.zeros((M, N + 1))
    L[:, 1:] = np.cumsum(dW, axis=1)
    
    return t, L

# ==========================================
# 3. Визуализация и сбор статистики
# ==========================================
def plot_and_analyze(t, L, alpha, beta):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'$\\alpha = {alpha}, \\beta = {beta}$', fontsize=14, fontweight='bold')
    
    # 1. Траектории (20 штук)
    ax1 = fig.add_subplot(2, 2, 1)
    for i in range(20):
        ax1.plot(t, L[i, :], lw=1, alpha=0.8)
    ax1.set_title('20 траекторий процесса')
    ax1.set_xlabel('Время (t)')
    ax1.set_ylabel('L(t)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Гистограмма распределения L(T)
    ax2 = fig.add_subplot(2, 2, 2)
    L_T = L[:, -1]
    
    # Ограничим выбросы для красивой гистограммы (отбрасываем 1% экстремумов)
    if alpha < 2.0:
        p1, p99 = np.percentile(L_T, [1, 99])
        L_T_plot = L_T[(L_T >= p1) & (L_T <= p99)]
    else:
        L_T_plot = L_T
        
    ax2.hist(L_T_plot, bins=50, density=True, color='orange', edgecolor='black', alpha=0.7)
    ax2.set_title('Распределение L(T) при T=1.0')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Эмпирическая дисперсия от времени
    ax3 = fig.add_subplot(2, 2, 3)
    emp_var = np.var(L, axis=0)
    ax3.plot(t, emp_var, 'r-', lw=2)
    title_var = 'Эмпирическая дисперсия от времени\n(Линейный рост)' if alpha == 2.0 else 'Эмпирическая дисперсия от времени\n(Отсутствие сходимости, скачки)'
    ax3.set_title(title_var)
    ax3.set_xlabel('Время (t)')
    ax3.set_ylabel('Дисперсия')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # 4. Q-Q Plot
    ax4 = fig.add_subplot(2, 2, 4)
    stats.probplot(L_T, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Сравнение с нормальным распр.)')
    ax4.get_lines()[0].set_marker('o')
    ax4.get_lines()[0].set_markersize(4)
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # --- Расчет статистики для Таблицы ---
    print(f"--- Статистика для alpha={alpha}, beta={beta} ---")
    print(f"Среднее: {np.mean(L_T):.4f}")
    print(f"Медиана: {np.median(L_T):.4f}")
    print(f"Дисперсия: {np.var(L_T):.4f}")
    print(f"Эксцесс: {stats.kurtosis(L_T):.4f}")
    print(f"95-й перц.: {np.percentile(L_T, 95):.4f}")
    print(f"99-й перц.: {np.percentile(L_T, 99):.4f}")
    print("-" * 40)

# ==========================================
# 4. Основной цикл запуска
# ==========================================
if __name__ == "__main__":
    # Новые параметры для вашей работы
    parameters = [
        (2.0, 0.0),   # Броуновское движение (база)
        (1.7, 0.0),   # Симметричные тяжелые хвосты
        (0.9, 0.0),   # Экстремально тяжелые хвосты (alpha < 1)
        (1.6, -0.7)   # Асимметрия с левым скосом (прыжки вниз)
    ]
    
    for alpha, beta in parameters:
        t, L = simulate_levy_process(alpha, beta, M=2000, N=1000, T=1.0)
        plot_and_analyze(t, L, alpha, beta)