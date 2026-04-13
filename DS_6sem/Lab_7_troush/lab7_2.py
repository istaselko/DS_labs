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
ticker = "ASML"
print(f"Загрузка данных для {ticker}...")
df = yf.Ticker(ticker).history(start="2021-01-01", end="2024-01-01")
data = df[["Close"]]
returns = np.log(data / data.shift(1)).dropna() * 100 
r_real = returns['Close']

# =====================================================================
# ЧАСТЬ 2: Симуляция GARCH(1,1), Оценка части ряда и Прогноз
# =====================================================================
print("\n=========================================================")
print("ЧАСТЬ 2: СИМУЛЯЦИЯ GARCH(1,1) И ОЦЕНКА ПАРАМЕТРОВ")
print("=========================================================")

def simulate_garch(omega, alpha, beta, noise, n=1000):
    r = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    r[0] = np.sqrt(sigma2[0]) * noise[0]

    for t in range(1, n):
        sigma2[t] = omega + alpha * (r[t - 1] ** 2) + beta * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * noise[t]
    return r, np.sqrt(sigma2)

true_omega, true_alpha, true_beta = 0.05, 0.1, 0.85
n_sim = 1000

np.random.seed(42)
noise_norm = np.random.normal(0, 1, n_sim)
noise_ts = stats.laplace.rvs(loc=0, scale=1 / np.sqrt(2), size=n_sim) 
noise_stable = stats.levy_stable.rvs(alpha=1.8, beta=0, loc=0, scale=0.5, size=n_sim)

r_norm, sig_norm = simulate_garch(true_omega, true_alpha, true_beta, noise_norm, n_sim)
r_ts, sig_ts = simulate_garch(true_omega, true_alpha, true_beta, noise_ts, n_sim)
r_stable, sig_stable = simulate_garch(true_omega, true_alpha, true_beta, noise_stable, n_sim)

def evaluate_and_plot_simulation(r_sim, sigma_true, name):
    train_size = int(len(r_sim) * 0.8) 
    r_train = r_sim[:train_size]
    
    am = arch_model(r_train, vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    
    print(f"\n--- Оценка симуляции с шумом: {name} ---")
    print(f"Истинные параметры: omega={true_omega}, alpha={true_alpha}, beta={true_beta}")
    print(f"Оцененные параметры: omega={res.params['omega']:.4f}, alpha={res.params['alpha[1]']:.4f}, beta={res.params['beta[1]']:.4f}")
    
    # ==== НОВОЕ: Расчет MSE для симуляции ====
    mse_train = np.mean((sigma_true[:train_size] - res.conditional_volatility)**2)
    print(f"MSE оценки волатильности (Train): {mse_train:.4f}")
    # =========================================

    raw_std_resid = res.resid / res.conditional_volatility
    std_resid = raw_std_resid[~np.isnan(raw_std_resid)] 
    
    if "Normal" in name:
        mu_est, std_est = stats.norm.fit(std_resid)
        print(f"Ошибка (Normal): mu = {mu_est:.4f}, std = {std_est:.4f}")
    elif "Laplace" in name:
        loc_est, scale_est = stats.laplace.fit(std_resid)
        print(f"Ошибка (TS/Laplace proxy): loc = {loc_est:.4f}, scale = {scale_est:.4f}")
    elif "Stable" in name:
        sample = np.random.choice(std_resid, min(300, len(std_resid)), replace=False)
        alpha_est, beta_est, loc_est, scale_est = stats.levy_stable.fit(sample)
        print(f"Ошибка (Stable): alpha = {alpha_est:.4f}, beta = {beta_est:.4f}")

    forecasts = res.forecast(horizon=len(r_sim) - train_size, reindex=False)
    pred_vol = np.sqrt(forecasts.variance.iloc[-1].values)
    
    plt.figure(figsize=(12, 4))
    plt.plot(sigma_true, label="Истинная волатильность", color='black', alpha=0.4, linewidth=2)
    plt.plot(range(train_size), res.conditional_volatility, label="Оцененная", color='blue', alpha=0.8)
    plt.plot(range(train_size, len(r_sim)), pred_vol, label="Прогноз", color='red', linestyle='--', linewidth=2)
    plt.axvline(x=train_size, color='green', linestyle=':', label='Train/Test split')
    plt.title(f"Симуляция GARCH(1,1) | Шум: {name}")
    plt.legend()
    plt.show()

evaluate_and_plot_simulation(r_norm, sig_norm, "Normal")
evaluate_and_plot_simulation(r_ts, sig_ts, "Tempered Stable proxy (Laplace)")
evaluate_and_plot_simulation(r_stable, sig_stable, "Stable")

# =====================================================================
# ЧАСТЬ 3: Оценка GARCH на РЕАЛЬНЫХ данных
# =====================================================================
print("\n=========================================================")
print(f"ЧАСТЬ 3: GARCH НА РЕАЛЬНЫХ ДАННЫХ ({ticker})")
print("=========================================================")

am = arch_model(r_real, vol="Garch", p=1, q=1, dist="normal")
res = am.fit(disp="off")

# ==== НОВОЕ: ВЫВОД ПАРАМЕТРОВ ДЛЯ ОТЧЕТА ====
print(f"Оцененные параметры модели GARCH(1,1) для {ticker}:")
print(f"omega = {res.params['omega']:.6f}, alpha = {res.params['alpha[1]']:.6f}, beta = {res.params['beta[1]']:.6f}")

# ==== НОВОЕ: Расчет MSE для реальных данных ====
mse_real = np.mean((np.abs(r_real) - res.conditional_volatility)**2)
print(f"MSE оценки волатильности на реальных данных (|r_t| vs Оцененная): {mse_real:.4f}")
# ===============================================

std_resid = (res.resid / res.conditional_volatility).dropna()

print("\n--- Оценка параметров распределений ошибки GARCH ---")
mu_n, std_n = stats.norm.fit(std_resid)
print(f"1. Normal -> mu: {mu_n:.4f}, std: {std_n:.4f}")

sample_st = np.random.choice(std_resid, min(300, len(std_resid)), replace=False)
alpha_st, beta_st, loc_st, scale_st = stats.levy_stable.fit(sample_st)
print(f"2. Stable -> alpha: {alpha_st:.4f}, beta: {beta_st:.4f}")
print("3. TS (Tweedie) -> Оценивается в среде R")

# =====================================================================
# ЧАСТЬ 4: Прогноз и График для реальных данных (Монте-Карло)
# =====================================================================
print("\n=========================================================")
print("ЧАСТЬ 4: ПРОГНОЗ ВОЛАТИЛЬНОСТИ ДЛЯ 3 РАСПРЕДЕЛЕНИЙ")
print("=========================================================")

horizon = 50
n_paths = 1000
last_var = res.conditional_volatility.iloc[-1]**2
last_ret = r_real.iloc[-1]

omega_e = res.params['omega']
alpha_e = res.params['alpha[1]']
beta_e  = res.params['beta[1]']

def mc_forecast(dist_type):
    np.random.seed(42)
    sim_vol = np.zeros((n_paths, horizon))
    for i in range(n_paths):
        if dist_type == "Normal":
            z = stats.norm.rvs(loc=mu_n, scale=std_n, size=horizon)
        elif dist_type == "TS (Laplace)":
            z = stats.laplace.rvs(loc=0, scale=1/np.sqrt(2), size=horizon)
        elif dist_type == "Stable":
            z = stats.levy_stable.rvs(alpha=alpha_st, beta=beta_st, loc=loc_st, scale=scale_st, size=horizon)
        
        curr_var = last_var
        curr_ret = last_ret
        for t in range(horizon):
            next_var = omega_e + alpha_e * (curr_ret**2) + beta_e * curr_var
            next_vol = np.sqrt(next_var)
            curr_ret = next_vol * z[t]
            curr_var = next_var
            sim_vol[i, t] = next_vol
            
    return np.median(sim_vol, axis=0)

future_dates = pd.bdate_range(start=r_real.index[-1], periods=horizon + 1)[1:]

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
dist_names = ["Normal", "TS (Laplace)", "Stable"]
colors = ["red", "purple", "darkorange"]

for i, dist in enumerate(dist_names):
    forecast_vol = mc_forecast(dist)
    axes[i].plot(r_real.index[-150:], np.abs(r_real)[-150:], color='gray', alpha=0.3, label='|r_t| (Прокси)')
    axes[i].plot(r_real.index[-150:], res.conditional_volatility[-150:], color='blue', label='Оцененная GARCH')
    axes[i].plot(future_dates, forecast_vol, color=colors[i], marker='.', linestyle='--', label=f'Прогноз ({dist})')
    
    axes[i].set_title(f"Прогноз волатильности ASML. Модель GARCH(1,1) + {dist} шум")
    axes[i].set_ylabel("Волатильность (%)")
    axes[i].legend(loc="upper left")
    axes[i].grid(True, alpha=0.3)

plt.xlabel("Дата")
plt.tight_layout()
plt.show()