import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from arch import arch_model

# 1. Загрузка данных 
tickers = ["ASML", "URA"]
start_date = "2021-01-01"
end_date = "2024-01-01"

print("Загрузка данных...")
data_dict = {}
for ticker in tickers:
    # Скачиваем историю для каждого тикера отдельно и берем цену закрытия
    df = yf.Ticker(ticker).history(start=start_date, end=end_date)
    data_dict[ticker] = df["Close"]

# Объединяем в одну таблицу
data = pd.DataFrame(data_dict)

# 2. Расчет дневных логарифмических доходностей
# r_t = ln(P_t / P_{t-1})
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
        "Эксцесс (Kurtosis)": returns.kurtosis(),  # Excess kurtosis (у нормального = 0)
        "Минимум": returns.min(),
        "Максимум": returns.max(),
    }
).T

print(stats_df.round(6))

# 4. Визуализация
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 4.1 Динамика цен
axes[0, 0].plot(data["ASML"], label="ASML", color="blue")
axes[0, 0].set_title("Динамика цен закрытия (ASML)")
axes[0, 0].set_ylabel("Цена (USD)")
axes[0, 0].tick_params(axis="x", rotation=45)

axes[0, 1].plot(data["URA"], label="URA", color="orange")
axes[0, 1].set_title("Динамика цен закрытия (URA)")
axes[0, 1].set_ylabel("Цена (USD)")
axes[0, 1].tick_params(axis="x", rotation=45)

# 4.2 Графики логарифмических доходностей (Волатильность)
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

# 5. Графики Box-Whisker (Ящик с усами) и Гистограммы
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot
sns.boxplot(data=returns, ax=axes[0], palette="Set2")
axes[0].set_title(
    "Box-Whisker график доходностей\n(Демонстрация выбросов/тяжелых хвостов)"
)
axes[0].set_ylabel("Доходность")

# Гистограммы
sns.histplot(
    returns["ASML"],
    kde=True,
    stat="density",
    color="blue",
    label="ASML",
    ax=axes[1],
    alpha=0.5,
)
sns.histplot(
    returns["URA"],
    kde=True,
    stat="density",
    color="orange",
    label="URA",
    ax=axes[1],
    alpha=0.5,
)
axes[1].set_title("Гистограммы распределений доходностей")
axes[1].set_xlabel("Доходность")
axes[1].legend()

plt.tight_layout()
plt.show()
