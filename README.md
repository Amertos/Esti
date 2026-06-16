# 📈 Esti AI Trading Bot
An end-to-end quantitative trading system that utilizes Machine Learning (XGBoost) to predict price movements of cryptocurrencies (BTC, ETH, SOL) and traditional assets (AAPL, TSLA, SPY).

## 🚀 Features
- **Data Engineering:** Automated historical data fetching (via `yfinance`) and calculation of over 30 technical indicators (SMA, MACD, RSI, Bollinger Bands, OBV, Volatility, etc.).
- **Machine Learning Pipeline:** Optimized `XGBoost` model configured with a `TimeSeriesSplit` cross-validation approach to prevent data leakage. Addresses class imbalance problems automatically using `scale_pos_weight`.
- **Real-Time Dashboard:** A premium, interactive Streamlit web application with Plotly charts. Features a modern "Glassmorphism" UI design, an AI probability gauge, and a live feature importance visualization.

## ⚙️ Project Architecture
The project is divided into 3 modular phases:
1. `phase1_data.py` - Fetches raw data, computes technical indicators, and creates the target variable (>2% growth in 3 days). Outputs a clean `.csv` dataset.
2. `phase2_model.py` - Trains the XGBoost model on historical data and tests it on unseen future data. Saves the trained model as a `.pkl` file and its parameters in `.json`.
3. `phase3_dashboard.py` - The UI. Fetches live market data, processes real-time indicators, and feeds them into the trained model to generate actionable trading signals and probabilities.

## 📦 Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/esti.git
cd esti
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the system (3-step pipeline)**
```bash
# 1. Generate the dataset
python phase1_data.py

# 2. Train the AI model
python phase2_model.py

# 3. Launch the Web Dashboard
streamlit run phase3_dashboard.py
```

## ⚠️ Disclaimer
This code is written strictly for educational purposes (Portfolio project) and to demonstrate Data Science and Machine Learning skills. ML models can make errors. **Do not use** this as financial advice.

## 👨‍💻 Tech Stack
* **Data Science:** Python, Pandas, Numpy
* **Machine Learning:** Scikit-Learn, XGBoost, TA (Technical Analysis library)
* **Visualization & UI:** Streamlit, Plotly, Custom CSS, HTML
* **Data Source:** Yahoo Finance API (yfinance)
