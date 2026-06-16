"""
PHASE 1: PRO DATA ENGINEERING PIPELINE
Skida 5 godina BTC podataka i kreira 30+ tehnickih indikatora.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings('ignore')


def fetch_data(symbol="BTC-USD", period="5y"):
    print(f"[FETCH] {symbol} | {period} istorije...")
    df = yf.download(symbol, period=period, interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"Nema podataka za {symbol}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    print(f"  Skinuto {len(df)} dana sirovih podataka.")
    return df


def add_indicators(df):
    print("[ENGINEERING] Pravljenje 30+ profesionalnih indikatora...")

    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    # --- TREND INDIKATORI ---
    for w in [10, 20, 50, 200]:
        df[f'SMA_{w}'] = close.rolling(w).mean()
        df[f'EMA_{w}'] = close.ewm(span=w, adjust=False).mean()

    # Golden / Death Cross signal (SMA 50 vs 200)
    df['GoldenCross'] = (df['SMA_50'] > df['SMA_200']).astype(int)

    # Cena relativno prema SMA50 i SMA200 (% koliko je iznad/ispod)
    df['Price_vs_SMA50']  = (close - df['SMA_50'])  / df['SMA_50']  * 100
    df['Price_vs_SMA200'] = (close - df['SMA_200']) / df['SMA_200'] * 100

    # --- MOMENTUM INDIKATORI ---
    df['RSI_14']  = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df['RSI_7']   = ta.momentum.RSIIndicator(close=close, window=7).rsi()

    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()

    df['ROC_5']  = ta.momentum.ROCIndicator(close=close, window=5).roc()
    df['ROC_10'] = ta.momentum.ROCIndicator(close=close, window=10).roc()

    # --- MACD ---
    macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df['MACD_Line']   = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff']   = macd.macd_diff()
    df['MACD_Cross']  = (df['MACD_Line'] > df['MACD_Signal']).astype(int)

    # --- VOLATILNOST ---
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df['BB_High']     = bb.bollinger_hband()
    df['BB_Low']      = bb.bollinger_lband()
    df['BB_Width']    = (df['BB_High'] - df['BB_Low']) / close  # Sirina kanala
    df['BB_Position'] = (close - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])

    df['ATR_14']        = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    df['Volatility_20'] = close.pct_change().rolling(20).std() * np.sqrt(365) * 100  # Anulizovana volatilnost

    # --- VOLUMEN ---
    df['OBV']          = ta.volume.OnBalanceVolumeIndicator(close=close, volume=vol).on_balance_volume()
    df['Volume_SMA20'] = vol.rolling(20).mean()
    df['Volume_Ratio'] = vol / df['Volume_SMA20']  # Da li je danas neobicno velik volumen?

    # --- POVRATI I ISTORIJA (LAG FEATURES - "Memorija" mašine) ---
    df['Daily_Return']   = close.pct_change() * 100
    df['Weekly_Return']  = close.pct_change(5) * 100
    df['Monthly_Return'] = close.pct_change(21) * 100

    for lag in [1, 2, 3, 5, 7, 14]:
        df[f'Return_Lag{lag}'] = df['Daily_Return'].shift(lag)

    # Dani u nedelji (ponedeljak=0, petak=4) - sezonalnost
    df['DayOfWeek'] = df.index.dayofweek

    return df


def create_target(df, threshold=0.02):
    """
    Target: Da li ce cena skociti vise od 2% u sledecih 3 dana?
    Gledamo 3 dana unapred jer je manje suma i trader ima vremena da reaguje.
    """
    print(f"[TARGET] Definisanje cilja: Skok >={threshold*100}% u sledeca 3 dana...")
    future_return = close_3d = df['Close'].shift(-3) / df['Close'] - 1
    df['Target'] = (future_return >= threshold).astype(int)
    return df


def run_pipeline(symbol="BTC-USD", period="5y"):
    print(f"\n{'='*55}")
    print(f" PRO DATA PIPELINE: {symbol}")
    print(f"{'='*55}")

    df = fetch_data(symbol, period)
    df = add_indicators(df)
    df = create_target(df, threshold=0.02)
    df.dropna(inplace=True)

    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(base_dir, f"{symbol.replace('-','_')}_dataset.csv")
    df.to_csv(output)

    # Statistika
    buy_pct = df['Target'].mean() * 100
    print(f"\n[DONE] {len(df)} redova | {len(df.columns)} kolona")
    print(f"  Kupi signala: {buy_pct:.1f}% dana | Prodaj/Cekaj: {100-buy_pct:.1f}%")
    print(f"  Sacuvano: {output}\n")
    return df


if __name__ == "__main__":
    for simbol in ["BTC-USD", "ETH-USD"]:
        run_pipeline(simbol, "5y")
