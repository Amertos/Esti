# 📈 Quant AI Trading Bot
End-to-End sistem za kvantitativno trgovanje koji koristi Machine Learning (XGBoost) za predviđanje kretanja cena kriptovaluta (BTC, ETH, SOL) i tradicionalnih aseta (AAPL, TSLA, SPY).

## 🚀 Mogućnosti (Features)
- **Data Engineering:** Automatsko preuzimanje istorijskih podataka (yfinance) i računanje preko 30 tehničkih indikatora (SMA, MACD, RSI, Bollinger Bands, OBV, Volatilnost).
- **Machine Learning Pipeline:** Optimizovan `XGBoost` model konfigurisan sa `TimeSeriesSplit` pristupom za evaluaciju bez "Data Leakage-a", kao i sa rešenim problemom disbalansa klasa (`scale_pos_weight`).
- **Real-Time Dashboard:** Premium Streamlit web aplikacija sa Plotly interaktivnim grafikonima. Implementiran je "Glassmorphism" dizajn, AI Gauge metar, i živa analiza feature importance-a (šta model posmatra).

## ⚙️ Arhitektura Projekta
Projekat je podeljen u 3 faze:
1. `phase1_data.py` - Prikuplja podatke, računa indikatore i kreira target varijablu (>2% rast u 3 dana). Zatim izbacuje `.csv` dataset.
2. `phase2_model.py` - Trenira model na prošlosti i testira ga na budućnosti. Snima gotov model kao `.pkl` i parametre u `.json`.
3. `phase3_dashboard.py` - UI. Preuzima žive (live) podatke, procesira indikatore i unosi ih u istrenirani model, koji potom generiše signal i verovatnoću kretanja tržišta.

## 📦 Instalacija i Pokretanje

**1. Kloniranje repozitorijuma**
```bash
git clone https://github.com/TvojUsername/quant-ai-bot.git
cd quant-ai-bot
```

**2. Instaliranje zavisnosti**
```bash
pip install -r requirements.txt
```

**3. Pokretanje sistema u 3 koraka**
```bash
# 1. Kreiranje podataka
python phase1_data.py

# 2. Trening Modela
python phase2_model.py

# 3. Pokretanje Dashboarda
streamlit run phase3_dashboard.py
```

## ⚠️ Disclaimer
Ovaj kod je napisan isključivo u edukativne svrhe (Portfolio projekat) i demonstraciju Data Science veština. Modeli mogu praviti greške. **Ne koristite** ovo kao finansijski savet.

## 👨‍💻 Tehnologije
* **Data Science:** Python, Pandas, Numpy
* **Machine Learning:** Scikit-Learn, XGBoost, TA (Technical Analysis)
* **Vizualizacija i UI:** Streamlit, Plotly, Custom CSS, HTML
* **Podaci:** Yahoo Finance (yfinance)
