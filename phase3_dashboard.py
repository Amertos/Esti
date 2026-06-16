import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle, json, ta, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Quant AI Bot", layout="wide", page_icon="📈")

CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
* { font-family: 'Inter', sans-serif !important; }
.stApp { background: radial-gradient(ellipse at 20% 50%, #0f1b2d 0%, #060d18 50%, #000308 100%); }
[data-testid="stSidebar"] { background: rgba(10,18,35,0.9) !important; backdrop-filter: blur(20px); border-right: 1px solid rgba(88,166,255,0.15) !important; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="metric-container"] { background: rgba(22,30,50,0.7) !important; backdrop-filter: blur(16px); border: 1px solid rgba(88,166,255,0.15) !important; border-radius: 16px !important; padding: 18px !important; box-shadow: 0 8px 32px rgba(0,0,0,0.4); transition: transform 0.3s ease, box-shadow 0.3s ease; }
[data-testid="metric-container"]:hover { transform: translateY(-4px); box-shadow: 0 20px 48px rgba(88,166,255,0.15); }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-weight: 700 !important; font-size: 1.5rem !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 1px; }
.stTabs [data-baseweb="tab-list"] { background: rgba(10,18,35,0.7) !important; backdrop-filter: blur(12px); border-radius: 12px !important; border: 1px solid rgba(88,166,255,0.1); padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #8b949e !important; border-radius: 8px !important; padding: 8px 20px !important; transition: all 0.2s ease; }
.stTabs [aria-selected="true"] { color: #e6edf3 !important; background: rgba(88,166,255,0.15) !important; border-bottom: none !important; }
h1 { background: linear-gradient(135deg, #58a6ff, #bc8cff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700 !important; }
h2, h3 { color: #c9d1d9 !important; font-weight: 600 !important; }
@keyframes pulse-buy { 0%,100% { box-shadow: 0 0 30px rgba(46,160,67,0.4); } 50% { box-shadow: 0 0 70px rgba(86,211,100,0.7); } }
@keyframes pulse-sell { 0%,100% { box-shadow: 0 0 30px rgba(248,81,73,0.4); } 50% { box-shadow: 0 0 70px rgba(255,123,114,0.7); } }
@keyframes pulse-hold { 0%,100% { box-shadow: 0 0 30px rgba(210,153,34,0.4); } 50% { box-shadow: 0 0 70px rgba(227,179,65,0.7); } }
.sig-buy { background: linear-gradient(135deg, rgba(13,79,53,0.85), rgba(26,122,80,0.65)); backdrop-filter: blur(20px); border: 1px solid rgba(46,160,67,0.7); border-radius: 20px; padding: 36px; text-align: center; font-size: 30px; font-weight: 800; color: #56d364; animation: pulse-buy 2s ease-in-out infinite; }
.sig-sell { background: linear-gradient(135deg, rgba(79,13,13,0.85), rgba(122,26,26,0.65)); backdrop-filter: blur(20px); border: 1px solid rgba(248,81,73,0.7); border-radius: 20px; padding: 36px; text-align: center; font-size: 30px; font-weight: 800; color: #ff7b72; animation: pulse-sell 2s ease-in-out infinite; }
.sig-hold { background: linear-gradient(135deg, rgba(50,45,0,0.85), rgba(80,70,0,0.65)); backdrop-filter: blur(20px); border: 1px solid rgba(210,153,34,0.7); border-radius: 20px; padding: 36px; text-align: center; font-size: 30px; font-weight: 800; color: #e3b341; animation: pulse-hold 2s ease-in-out infinite; }
.ind-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid rgba(88,166,255,0.08); }
.ind-label { color: #8b949e; font-size: 0.85rem; }
.ind-val { font-weight: 600; font-size: 0.95rem; }
.bull { color: #56d364; } .bear { color: #ff7b72; } .neut { color: #e3b341; }
.stButton > button { background: linear-gradient(135deg, rgba(88,166,255,0.15), rgba(188,140,255,0.1)) !important; border: 1px solid rgba(88,166,255,0.3) !important; color: #58a6ff !important; border-radius: 10px !important; font-weight: 600 !important; transition: all 0.3s ease !important; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(88,166,255,0.25) !important; }
[data-testid="stSelectbox"] > div > div { background: rgba(22,30,50,0.8) !important; border: 1px solid rgba(88,166,255,0.2) !important; border-radius: 10px !important; }
::-webkit-scrollbar { width: 5px; } ::-webkit-scrollbar-track { background: rgba(10,18,35,0.5); } ::-webkit-scrollbar-thumb { background: rgba(88,166,255,0.3); border-radius: 3px; }
hr { border-color: rgba(88,166,255,0.1) !important; margin: 20px 0 !important; }
@keyframes fadein { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
.main > div { animation: fadein 0.5s ease-out; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

import os
BASE = os.path.dirname(os.path.abspath(__file__))
SYMBOLS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "S&P 500 (SPY)": "SPY",
}
DROP_COLS = ['Target','Close','Open','High','Low','Volume','BB_High','BB_Low','SMA_200','EMA_200','Volume_SMA20']

# ── LOADERS ────────────────────────────────────────
@st.cache_resource
def load_model(sym):
    try:
        with open(f"{BASE}/{sym.replace('-','_')}_model.pkl",'rb') as f:
            return pickle.load(f)
    except: return None

@st.cache_resource
def load_meta(sym):
    try:
        with open(f"{BASE}/{sym.replace('-','_')}_meta.json") as f:
            return json.load(f)
    except: return None

@st.cache_data(ttl=1800)
def get_data(sym):
    df = yf.download(sym, period="2y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    c,h,l,v = df['Close'], df['High'], df['Low'], df['Volume']
    for w in [10,20,50,200]:
        df[f'SMA_{w}'] = c.rolling(w).mean()
        df[f'EMA_{w}'] = c.ewm(span=w, adjust=False).mean()
    df['GoldenCross']     = (df['SMA_50'] > df['SMA_200']).astype(int)
    df['Price_vs_SMA50']  = (c - df['SMA_50'])  / df['SMA_50']  * 100
    df['Price_vs_SMA200'] = (c - df['SMA_200']) / df['SMA_200'] * 100
    df['RSI_14']   = ta.momentum.RSIIndicator(close=c, window=14).rsi()
    df['RSI_7']    = ta.momentum.RSIIndicator(close=c, window=7).rsi()
    stoch = ta.momentum.StochasticOscillator(high=h, low=l, close=c)
    df['Stoch_K']  = stoch.stoch()
    df['Stoch_D']  = stoch.stoch_signal()
    df['ROC_5']    = ta.momentum.ROCIndicator(close=c, window=5).roc()
    df['ROC_10']   = ta.momentum.ROCIndicator(close=c, window=10).roc()
    macd = ta.trend.MACD(close=c)
    df['MACD_Line']   = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff']   = macd.macd_diff()
    df['MACD_Cross']  = (df['MACD_Line'] > df['MACD_Signal']).astype(int)
    bb = ta.volatility.BollingerBands(close=c)
    df['BB_High']     = bb.bollinger_hband()
    df['BB_Low']      = bb.bollinger_lband()
    df['BB_Width']    = (df['BB_High'] - df['BB_Low']) / c
    df['BB_Position'] = (c - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])
    df['ATR_14']      = ta.volatility.AverageTrueRange(high=h, low=l, close=c).average_true_range()
    df['Volatility_20'] = c.pct_change().rolling(20).std() * np.sqrt(365) * 100
    df['OBV']           = ta.volume.OnBalanceVolumeIndicator(close=c, volume=v).on_balance_volume()
    df['Volume_SMA20']  = v.rolling(20).mean()
    df['Volume_Ratio']  = v / df['Volume_SMA20']
    df['Daily_Return']  = c.pct_change() * 100
    df['Weekly_Return'] = c.pct_change(5) * 100
    df['Monthly_Return']= c.pct_change(21) * 100
    for lag in [1,2,3,5,7,14]:
        df[f'Return_Lag{lag}'] = df['Daily_Return'].shift(lag)
    df['DayOfWeek'] = df.index.dayofweek
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# ── SIDEBAR ────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>⚡ QUANT ENGINE</h2>", unsafe_allow_html=True)
    st.markdown("---")
    selected_name = st.selectbox("Izaberi Asset", list(SYMBOLS.keys()))
    symbol = SYMBOLS[selected_name]
    chart_period = st.selectbox("Period Analize", ["3mo", "6mo", "1y", "2y"], index=2)
    st.markdown("---")
    st.markdown("### 🎛️ HUD Kontrole")
    show_sma   = st.checkbox("Prikazi SMA (10/50/200)", value=True)
    show_bb    = st.checkbox("Bollinger Bands", value=True)
    show_vol   = st.checkbox("Volumen", value=True)
    show_macd  = st.checkbox("MACD Oscilator", value=True)
    show_rsi   = st.checkbox("RSI Momentum", value=True)
    st.markdown("---")
    if st.button("🔄 Osvezi Live Podatke", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── CORE LOGIC ─────────────────────────────────────
model = load_model(symbol)
meta  = load_meta(symbol)

with st.spinner(f"Skidanje i obrada {symbol} podataka..."):
    df = get_data(symbol)
    period_map = {"3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
    cutoff = df.index[-1] - pd.Timedelta(days=period_map.get(chart_period, 365))
    live_chart = df[df.index >= cutoff]

last = df.iloc[-1]
prev = df.iloc[-2]
cur_price = float(last['Close'])
chg_pct = (cur_price - float(prev['Close'])) / float(prev['Close']) * 100

# ── MAIN HEADER ────────────────────────────────────
st.markdown(f"<h1>{selected_name} <span style='font-size: 1.5rem; color: #8b949e; font-weight:400;'>{symbol}</span></h1>", unsafe_allow_html=True)

# ── METRICS HUD ────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Live Cena", f"${cur_price:,.2f}", f"{chg_pct:+.2f}%")
m2.metric("RSI (14d)", f"{last['RSI_14']:.1f}", "OVERBOUGHT" if last['RSI_14']>70 else ("OVERSOLD" if last['RSI_14']<30 else "NEUTRAL"))
m3.metric("MACD Cross", "BULLISH" if last['MACD_Cross']==1 else "BEARISH", f"{last['MACD_Diff']:.2f}")
m4.metric("Volatilnost", f"{last['Volatility_20']:.1f}%")
m5.metric("Volumen", f"{last['Volume_Ratio']:.2f}x", "VISOK" if last['Volume_Ratio']>1.5 else "NORMALAN")
m6.metric("Trend", "GOLDEN CROSS" if last['GoldenCross']==1 else "DEATH CROSS")

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["🚀 Terminal (Grafikon)", "🧠 AI Quantum Signal", "🔬 Tehnicka Inspekcija", "🗄️ Raw Data"])

# ── TAB 1: GRAFIKON ────────────────────────────────
with t1:
    rows = 1
    heights = [0.5]
    titles = ["Price Action"]
    if show_vol:  rows+=1; heights.append(0.15); titles.append("Volume")
    if show_rsi:  rows+=1; heights.append(0.15); titles.append("RSI")
    if show_macd: rows+=1; heights.append(0.2);  titles.append("MACD")

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.03, subplot_titles=titles)

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=live_chart.index, open=live_chart['Open'], high=live_chart['High'],
        low=live_chart['Low'], close=live_chart['Close'], name="Cena",
        increasing_line_color='#56d364', decreasing_line_color='#f85149',
        increasing_fillcolor='#56d364', decreasing_fillcolor='#f85149'
    ), row=1, col=1)

    # Overlays
    if show_sma:
        fig.add_trace(go.Scatter(x=live_chart.index, y=live_chart['SMA_10'], line=dict(color='#58a6ff', width=1, dash='dot'), name="SMA10"), row=1, col=1)
        fig.add_trace(go.Scatter(x=live_chart.index, y=live_chart['SMA_50'], line=dict(color='#e3b341', width=1.5), name="SMA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=live_chart.index, y=live_chart['SMA_200'], line=dict(color='#ff7b72', width=2), name="SMA200"), row=1, col=1)
    
    if show_bb:
        fig.add_trace(go.Scatter(x=live_chart.index, y=live_chart['BB_High'], line=dict(color='rgba(88,166,255,0.2)', width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=live_chart.index, y=live_chart['BB_Low'], fill='tonexty', fillcolor='rgba(88,166,255,0.05)', line=dict(color='rgba(88,166,255,0.2)', width=1), name='Bollinger'), row=1, col=1)

    curr = 2
    if show_vol:
        v_colors = ['#56d364' if c>=o else '#f85149' for c,o in zip(live_chart['Close'], live_chart['Open'])]
        fig.add_trace(go.Bar(x=live_chart.index, y=live_chart['Volume'], marker_color=v_colors, name="Vol", opacity=0.6), row=curr, col=1)
        curr+=1

    if show_rsi:
        fig.add_trace(go.Scatter(x=live_chart.index, y=live_chart['RSI_14'], line=dict(color='#bc8cff', width=1.5), name='RSI'), row=curr, col=1)
        fig.add_hline(y=70, line=dict(color='#f85149', dash='dash', width=1), row=curr, col=1)
        fig.add_hline(y=30, line=dict(color='#56d364', dash='dash', width=1), row=curr, col=1)
        curr+=1

    if show_macd:
        fig.add_trace(go.Scatter(x=live_chart.index, y=live_chart['MACD_Line'], line=dict(color='#58a6ff', width=1.5), name='MACD'), row=curr, col=1)
        fig.add_trace(go.Scatter(x=live_chart.index, y=live_chart['MACD_Signal'], line=dict(color='#e3b341', width=1), name='Signal'), row=curr, col=1)
        m_colors = ['#56d364' if v>=0 else '#f85149' for v in live_chart['MACD_Diff']]
        fig.add_trace(go.Bar(x=live_chart.index, y=live_chart['MACD_Diff'], marker_color=m_colors, name='Hist'), row=curr, col=1)

    fig.update_layout(
        height=800, template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=10, r=10, t=40, b=10), font=dict(color='#c9d1d9')
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(88,166,255,0.05)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(88,166,255,0.05)')
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: AI SIGNAL ───────────────────────────────
with t2:
    if model is None or meta is None:
        st.error("⚠️ AI Model za ovaj asset nije pronađen. Molimo pokrenite Phase 2 trening.")
    else:
        # Priprema
        X_today = df.iloc[-1:].copy()
        for col in DROP_COLS:
            if col in X_today.columns: X_today = X_today.drop(columns=[col])
        m_features = list(model.feature_names_in_)
        X_today = X_today[[f for f in m_features if f in X_today.columns]]
        for f in m_features:
            if f not in X_today.columns: X_today[f] = 0
        X_today = X_today[m_features]

        # Inerencija
        prob_buy = float(model.predict_proba(X_today)[0][1])
        prob_sell = 1 - prob_buy

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("<h3 style='text-align:center;'>Projekcija: Naredna 3 Dana</h3>", unsafe_allow_html=True)
            if prob_buy >= 0.60:
                st.markdown(f"<div class='sig-buy'>🟢 LONG / BUY<br><span style='font-size:18px; font-weight:500; color:#c9d1d9;'>Pouzdanost Rasta: {prob_buy*100:.1f}%</span></div>", unsafe_allow_html=True)
            elif prob_buy >= 0.45:
                st.markdown(f"<div class='sig-hold'>🟡 HOLD / NEUTRAL<br><span style='font-size:18px; font-weight:500; color:#c9d1d9;'>Nesigurno trzište: {prob_buy*100:.1f}%</span></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='sig-sell'>🔴 SHORT / AVOID<br><span style='font-size:18px; font-weight:500; color:#c9d1d9;'>Rizik Pada: {prob_sell*100:.1f}%</span></div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_buy * 100,
                number={'suffix': '%', 'font': {'size': 50, 'color': '#e6edf3'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#8b949e'},
                    'bar': {'color': '#58a6ff', 'thickness': 0.25},
                    'bgcolor': 'rgba(22,30,50,0.5)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 45], 'color': 'rgba(248,81,73,0.2)'},
                        {'range': [45, 60], 'color': 'rgba(210,153,34,0.2)'},
                        {'range': [60, 100], 'color': 'rgba(46,160,67,0.2)'},
                    ],
                    'threshold': {'line': {'color': '#ffffff', 'width': 3}, 'value': 60}
                }
            ))
            fig_g.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e'))
            st.plotly_chart(fig_g, use_container_width=True)

        st.markdown("---")
        # Feature Importance
        st.markdown("### Sta AI trenutno gleda? (Feature Importance)")
        imp = pd.Series(model.feature_importances_, index=m_features).sort_values(ascending=True).tail(12)
        fig_imp = go.Figure(go.Bar(
            x=imp.values, y=imp.index, orientation='h',
            marker=dict(color=imp.values, colorscale='Blues', showscale=False)
        ))
        fig_imp.update_layout(
            height=350, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(color='#c9d1d9', tickfont=dict(size=12))
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ── TAB 3: INSPEKCIJA ──────────────────────────────
with t3:
    i1, i2, i3 = st.columns(3)
    
    with i1:
        st.markdown("### Momentum")
        def row(lbl, val, cls=""): return f"<div class='ind-row'><span class='ind-label'>{lbl}</span><span class='ind-val {cls}'>{val}</span></div>"
        
        rsi_cls = "bear" if last['RSI_14']>70 else ("bull" if last['RSI_14']<30 else "neut")
        st.markdown(row("RSI (14d)", f"{last['RSI_14']:.1f}", rsi_cls), unsafe_allow_html=True)
        st.markdown(row("Stoch K", f"{last['Stoch_K']:.1f}"), unsafe_allow_html=True)
        st.markdown(row("Stoch D", f"{last['Stoch_D']:.1f}"), unsafe_allow_html=True)
        
        roc_cls = "bull" if last['ROC_5']>0 else "bear"
        st.markdown(row("Rate of Change (5d)", f"{last['ROC_5']:+.2f}%", roc_cls), unsafe_allow_html=True)
        
    with i2:
        st.markdown("### Trend & Volatilnost")
        st.markdown(row("Od SMA50", f"{last['Price_vs_SMA50']:+.2f}%", "bull" if last['Price_vs_SMA50']>0 else "bear"), unsafe_allow_html=True)
        st.markdown(row("Od SMA200", f"{last['Price_vs_SMA200']:+.2f}%", "bull" if last['Price_vs_SMA200']>0 else "bear"), unsafe_allow_html=True)
        
        bb_pos = last['BB_Position']
        bb_cls = "bear" if bb_pos>0.9 else ("bull" if bb_pos<0.1 else "neut")
        st.markdown(row("BB Pozicija (0-1)", f"{bb_pos:.2f}", bb_cls), unsafe_allow_html=True)
        st.markdown(row("Volatilnost (Annu.)", f"{last['Volatility_20']:.1f}%"), unsafe_allow_html=True)

    with i3:
        st.markdown("### Volume & Patterni")
        vr = last['Volume_Ratio']
        vr_cls = "bull" if vr>1.5 else "neut"
        st.markdown(row("Volume Skok", f"{vr:.2f}x", vr_cls), unsafe_allow_html=True)
        st.markdown(row("Dnevni Povrat", f"{last['Daily_Return']:+.2f}%", "bull" if last['Daily_Return']>0 else "bear"), unsafe_allow_html=True)
        st.markdown(row("Nedeljni Povrat", f"{last['Weekly_Return']:+.2f}%", "bull" if last['Weekly_Return']>0 else "bear"), unsafe_allow_html=True)
        st.markdown(row("Mesecni Povrat", f"{last['Monthly_Return']:+.2f}%", "bull" if last['Monthly_Return']>0 else "bear"), unsafe_allow_html=True)

# ── TAB 4: DATA ────────────────────────────────────
with t4:
    st.markdown("### Raw Time-Series Podaci")
    disp = ['Close', 'Volume', 'RSI_14', 'MACD_Diff', 'BB_Position', 'Daily_Return']
    st.dataframe(
        df[disp].tail(50).sort_index(ascending=False).style.format({
            'Close':'${:,.2f}', 'Volume':'{:,.0f}', 'RSI_14':'{:.1f}',
            'MACD_Diff':'{:.2f}', 'BB_Position':'{:.2f}', 'Daily_Return':'{:+.2f}%'
        }),
        height=500, use_container_width=True
    )
    if st.button("💾 Download CSV Dump"):
        st.download_button("Klikni za Download", df.to_csv(), f"{symbol}_dump.csv", "text/csv")
