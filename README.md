<div align="center">
  <img src="https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&q=80&w=1200&h=300" alt="Esti Header" style="border-radius: 15px; margin-bottom: 20px;">

  <h1>📈 Esti: AI Quantitative Trading System</h1>
  <p><strong>Advanced Machine Learning Trading Bot & Real-Time Analytics Dashboard</strong></p>

  [![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://esti-finance-ai.streamlit.app/)
  [![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
  [![XGBoost](https://img.shields.io/badge/AI_Model-XGBoost-1273C4?style=for-the-badge)]()
</div>

<br>

## 📖 About the Project
**Esti** is a professional, end-to-end quantitative trading system designed to predict the price movements of cryptocurrencies and major tech stocks. It employs **Machine Learning (XGBoost)** to process decades of historical market data and generates real-time actionable signals (BUY/SELL/HOLD).

The project features a sleek, "Glassmorphism" styled Streamlit dashboard, providing investors with deep technical insights, real-time momentum tracking, and Monte Carlo probability forecasts.

---

## ⚡ Key Features

* **🧠 AI Quantum Signal:** XGBoost classifier predicting if an asset will grow >2% in the next 3 days, trained with strict `TimeSeriesSplit` cross-validation.
* **📊 30+ Technical Indicators:** Automated calculation of RSI, MACD, Stochastic Oscillators, Bollinger Bands, Volatility, and Lagged Returns.
* **🎲 Monte Carlo Simulation:** Advanced mathematical modeling generating 100 possible future price paths for the next 7 days based on recent volatility.
* **📈 Real-Time Interactive HUD:** Dynamic Plotly charts with togglable Overlays (Support & Resistance, SMA, Bollinger Bands, Volume).
* **🌐 Production Ready:** Fully portable codebase, GitHub-ready `.gitignore`, and seamless CI/CD integration with Streamlit Community Cloud.

---

## 🏗️ System Architecture

```mermaid
graph TD;
    A[Yahoo Finance API] -->|Historical Data| B(Phase 1: Data Engineering)
    B -->|30+ Technical Indicators| C{Phase 2: ML Pipeline}
    C -->|Train XGBoost Model| D[Phase 3: Real-Time Dashboard]
    D -->|Inference & Prediction| E[Live Trading Signals]
    C -->|Metadata & Weights| D
    A -->|Live Price Action| D
```

---

## 💻 Tech Stack

| Category | Technology |
| --- | --- |
| **Data Engineering** | `Pandas`, `Numpy`, `yfinance` |
| **Machine Learning** | `Scikit-Learn`, `XGBoost`, `ta` (Technical Analysis) |
| **Frontend / UI** | `Streamlit`, Custom CSS (Glassmorphism) |
| **Data Visualization** | `Plotly Graph Objects` |

---

## 🚀 Installation & Setup

If you want to run this project locally on your machine, follow these steps:

**1. Clone the repository:**
```bash
git clone https://github.com/Amertos/Esti.git
cd Esti
```

**2. Set up virtual environment and install dependencies:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

**3. Run the complete pipeline:**
```bash
# Generate the datasets
python phase1_data.py

# Train the AI Models
python phase2_model.py

# Launch the Web Application
streamlit run phase3_dashboard.py
```

---

## 📸 Dashboard Preview

> **Note to developer:** *Add a screenshot of your beautiful Streamlit dashboard here by uploading an image into your GitHub repository and linking it below!*
> 
> `![Dashboard Screenshot](link_to_your_image.png)`

---

## ⚠️ Disclaimer
This system is an **educational project** and a demonstration of Data Science, Machine Learning, and Software Engineering skills. It is **not** financial advice. Always do your own research before investing in the stock or crypto market. ML models can make mistakes.
