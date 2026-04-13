import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

# Отключаем предупреждения от KPSS
warnings.filterwarnings("ignore")

# 1. Загрузка данных 
tickers = ["ASML", "URA"]
start_date = "2021-01-01"
end_date = "2024-01-01"

print("Загрузка данных...")
data_dict = {}
for ticker in tickers:
    df = yf.Ticker(ticker).history(start=start_date, end=end_date)
    data_dict[ticker] = df["Close"]

data = pd.DataFrame(data_dict)

# 2. Расчет дневных логарифмических доходностей
returns = np.log(data / data.shift(1)).dropna()

# 3. Описательные статистики
print("\n--- Описательные статистики дневных доходностей ---")
stats_df = pd.DataFrame(
    {
        "Среднее (Mean)": returns.mean(),
        "Медиана (Median)": returns.median(),
        "Ст. отклонение (Std)": returns.std(),
        "Дисперсия (Var)": returns.var(),
        "Асимметрия (Skewness)": returns.skew(),
        "Эксцесс (Kurtosis)": returns.kurtosis(),
        "Минимум": returns.min(),
        "Максимум": returns.max(),
    }
).T
print(stats_df.round(6))

# ================= НОВОЕ: Тесты на стационарность =================
print("\n--- Тесты на стационарность (ADF и KPSS) ---")
for ticker in tickers:
    print(f"\nАнализ для {ticker}:")
    
    # Расширенный тест Дики-Фуллера (ADF)
    adf_result = adfuller(returns[ticker])
    print(f"ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
    if adf_result[1] < 0.05:
        print("  -> ADF: Отвергаем H0, ряд СТАЦИОНАРЕН")
    else:
        print("  -> ADF: Нет оснований отвергнуть H0, ряд НЕстационарен")

    # Тест KPSS
    kpss_result = kpss(returns[ticker], regression='c', nlags='auto')
    print(f"KPSS Statistic: {kpss_result[0]:.4f}, p-value: {kpss_result[1]:.4f}")
    if kpss_result[1] < 0.05:
        print("  -> KPSS: Отвергаем H0, ряд НЕстационарен")
    else:
        print("  -> KPSS: Нет оснований отвергнуть H0, ряд СТАЦИОНАРЕН")
# ==================================================================

# 4. Визуализация
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0, 0].plot(data["ASML"], label="ASML", color="blue")
axes[0, 0].set_title("Динамика цен закрытия (ASML)")
axes[0, 0].set_ylabel("Цена (USD)")
axes[0, 0].tick_params(axis="x", rotation=45)

axes[0, 1].plot(data["URA"], label="URA", color="orange")
axes[0, 1].set_title("Динамика цен закрытия (URA)")
axes[0, 1].set_ylabel("Цена (USD)")
axes[0, 1].tick_params(axis="x", rotation=45)

axes[1, 0].plot(returns["ASML"], color="blue", alpha=0.7)
axes[1, 0].set_title("Лог-доходности ASML (Кластеризация волатильности)")
axes[1, 0].set_ylabel("Доходность")
axes[1, 0].tick_params(axis="x", rotation=45)

axes[1, 1].plot(returns["URA"], color="orange", alpha=0.7)
axes[1, 1].set_title("Лог-доходности URA (Кластеризация волатильности)")
axes[1, 1].set_ylabel("Доходность")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# 5. Графики Box-Whisker и Гистограммы
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(data=returns, ax=axes[0], palette="Set2")
axes[0].set_title("Box-Whisker график доходностей\n(Демонстрация выбросов/тяжелых хвостов)")
axes[0].set_ylabel("Доходность")

sns.histplot(returns["ASML"], kde=True, stat="density", color="blue", label="ASML", ax=axes[1], alpha=0.5)
sns.histplot(returns["URA"], kde=True, stat="density", color="orange", label="URA", ax=axes[1], alpha=0.5)
axes[1].set_title("Гистограммы распределений доходностей")
axes[1].set_xlabel("Доходность")
axes[1].legend()

plt.tight_layout()
plt.show()