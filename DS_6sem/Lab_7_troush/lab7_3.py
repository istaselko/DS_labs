import numpy as np
import pandas as pd
from arch import arch_model
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# 1. Загрузка данных
tickers =["ASML", "URA"]
data_dict = {}
for ticker in tickers:
    df = yf.Ticker(ticker).history(start="2021-01-01", end="2024-01-01")
    data_dict[ticker] = df["Close"]
data = pd.DataFrame(data_dict)
returns = np.log(data / data.shift(1)).dropna() * 100

# 2. Оценка GARCH и экспорт остатков
def export_residuals(asset_name, r):
    am = arch_model(r, vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    std_resid = (res.resid / res.conditional_volatility).dropna()
    
    # Сохраняем в CSV файл
    filename = f"resid_{asset_name}.csv"
    std_resid.to_csv(filename, index=False, header=["resid"])
    print(f"Остатки для {asset_name} успешно сохранены в файл: {filename}")

export_residuals("ASML", returns["ASML"])
export_residuals("URA", returns["URA"])