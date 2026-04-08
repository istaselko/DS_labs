import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from arch import arch_model
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# =====================================================================
# ЧАСТЬ 1: Загрузка данных
# =====================================================================
tickers = ["ASML", "URA"]
data_dict = {}
for ticker in tickers:
    df = yf.Ticker(ticker).history(start="2021-01-01", end="2024-01-01")
    data_dict[ticker] = df["Close"]
data = pd.DataFrame(data_dict)
returns = np.log(data / data.shift(1)).dropna() * 100  # В процентах

# =====================================================================
# ЧАСТЬ 2: Моделирование GARCH(1,1) и проверка точности (Симоляция)
# =====================================================================
print("=========================================================")
print("ЧАСТЬ 2: МОДЕЛИРОВАНИЕ GARCH(1,1) И ОЦЕНКА ПАРАМЕТРОВ")
print("=========================================================")


def simulate_garch(omega, alpha, beta, noise, n=1000):
    """Генерация процесса GARCH(1,1)"""
    r = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    r[0] = np.sqrt(sigma2[0]) * noise[0]

    for t in range(1, n):
        sigma2[t] = omega + alpha * (r[t - 1] ** 2) + beta * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * noise[t]
    return r, np.sqrt(sigma2)


# Истинные параметры модели
true_omega, true_alpha, true_beta = 0.05, 0.1, 0.85
n_sim = 1500

# 1. Нормальный шум
noise_norm = np.random.normal(0, 1, n_sim)
r_norm, _ = simulate_garch(true_omega, true_alpha, true_beta, noise_norm)

# 2. Устойчивый шум (Stable) - alpha=1.7, beta=0 (масштабируем, чтобы дисперсия была ~1)
noise_stable = stats.levy_stable.rvs(alpha=1.7, beta=0, loc=0, scale=0.5, size=n_sim)
r_stable, _ = simulate_garch(true_omega, true_alpha, true_beta, noise_stable)

# 3. TS (Tempered Stable) шум (Используем распределение Лапласа как прокси-TS с тяжелыми хвостами)
noise_ts = stats.laplace.rvs(loc=0, scale=1 / np.sqrt(2), size=n_sim)
r_ts, _ = simulate_garch(true_omega, true_alpha, true_beta, noise_ts)


def estimate_garch_sim(r_sim, name):
    am = arch_model(r_sim, vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    print(f"\n--- Оценка симуляции с шумом: {name} ---")
    print(
        f"Истинные параметры: omega={true_omega}, alpha={true_alpha}, beta={true_beta}"
    )
    print(
        f"Оцененные параметры: omega={res.params['omega']:.4f}, alpha={res.params['alpha[1]']:.4f}, beta={res.params['beta[1]']:.4f}"
    )
    print(f"Критерий Акаике (AIC): {res.aic:.2f}")


estimate_garch_sim(r_norm, "Normal")
estimate_garch_sim(r_ts, "Tempered Stable (TS)")
estimate_garch_sim(r_stable, "Stable")

# =====================================================================
# ЧАСТЬ 3: Оценка GARCH(1,1) на РЕАЛЬНЫХ данных + Волатильность (3 случая)
# =====================================================================
print("\n=========================================================")
print("ЧАСТЬ 3: GARCH НА РЕАЛЬНЫХ ДАННЫХ И ВОЛАТИЛЬНОСТЬ")
print("=========================================================")


def analyze_real_garch(asset_name, r):
    print(f"\nАнализ актива: {asset_name}")
    am = arch_model(r, vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off")

    std_resid = (res.resid / res.conditional_volatility).dropna()

    # 1. Normal Fit
    mu_n, std_n = stats.norm.fit(std_resid)
    ks_norm = stats.kstest(std_resid, "norm", args=(mu_n, std_n))

    # 2. Stable Fit (Чтобы не зависло, берем 300 случайных точек)
    sample_st = np.random.choice(std_resid, min(300, len(std_resid)), replace=False)
    alpha_st, beta_st, loc_st, scale_st = stats.levy_stable.fit(sample_st)
    ks_stable = stats.kstest(
        std_resid, "levy_stable", args=(alpha_st, beta_st, loc_st, scale_st)
    )

    # 3. TS Fit (Используем Стьюдента / Обобщенное как аппроксимацию TS для быстроты и избежания зависаний)
    df_ts, loc_ts, scale_ts = stats.t.fit(std_resid)
    ks_ts = stats.kstest(std_resid, "t", args=(df_ts, loc_ts, scale_ts))

    print(f"Критерий Колмогорова-Смирнова (меньше статистика D = лучше):")
    print(f"  1. Normal: D={ks_norm.statistic:.4f}, p-value={ks_norm.pvalue:.4f}")
    print(f"  2. TS (proxy): D={ks_ts.statistic:.4f}, p-value={ks_ts.pvalue:.4f}")
    print(f"  3. Stable: D={ks_stable.statistic:.4f}, p-value={ks_stable.pvalue:.4f}")

    # Прогноз волатильности
    forecasts = res.forecast(horizon=10, reindex=False)
    pred_vol = np.sqrt(forecasts.variance.iloc[-1].values)

    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Анализ GARCH(1,1) для {asset_name}", fontsize=14)

    # График 1: Условная волатильность
    axes[0].plot(r.index, r, color="gray", alpha=0.4, label="Доходности")
    axes[0].plot(
        r.index, res.conditional_volatility, color="red", label="Волатильность"
    )
    axes[0].set_title("Оцененная волатильность (GARCH)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend()

    # График 2: Три случая распределений (Normal, TS, Stable)
    x = np.linspace(std_resid.min(), std_resid.max(), 500)
    axes[1].hist(
        std_resid, bins=40, density=True, alpha=0.4, color="blue", label="Остатки"
    )
    axes[1].plot(x, stats.norm.pdf(x, mu_n, std_n), "r-", label="1. Normal")
    axes[1].plot(
        x, stats.t.pdf(x, df_ts, loc_ts, scale_ts), "g--", label="2. TS (Heavy Tails)"
    )
    axes[1].plot(
        x,
        stats.levy_stable.pdf(x, alpha_st, beta_st, loc_st, scale_st),
        "k-.",
        label=f"3. Stable (a={alpha_st:.2f})",
    )
    axes[1].set_title("Сравнение 3-х случаев распределений")
    axes[1].set_xlim(-4, 4)
    axes[1].legend()

    # График 3: Прогноз волатильности
    axes[2].plot(range(1, 11), pred_vol, marker="o", color="purple")
    axes[2].set_title("Прогноз волатильности (10 дней)")
    axes[2].set_xlabel("Дни")

    plt.tight_layout()
    plt.show()


# Запускаем анализ для обоих активов
analyze_real_garch("ASML", returns["ASML"])
analyze_real_garch("URA", returns["URA"])
