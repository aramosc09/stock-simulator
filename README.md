# Stock Visualizer + Basic Strategy Simulator

This project was developed as part of a class at **Tecnológico de Monterrey**, aiming to demonstrate core concepts of:

- **Streamlit** for building interactive web applications in Python.
- **Data visualization** using Plotly and Streamlit components.
- **Basic investment strategy simulation** using historical stock data and backtesting with Backtrader.

---

## 🔍 What does the app do?

The app allows users to:

1. **Search and visualize** stock data from Yahoo Finance.
2. **Analyze historical price trends** using Bollinger Bands and Simple Moving Averages (SMA).
3. **Design a basic buy/sell investment strategy**, adjusting parameters like:
   - Initial capital
   - Risk (buy aggressiveness)
   - Panic level (sell threshold)
   - Estimated inflation for adjusted return
4. **Backtest** their strategy over a selected historical period.
5. **Evaluate key metrics** like return, drawdown, Sharpe Ratio, win rate, and more.
6. **Download raw stock data** in CSV format.
7. **Stay updated** with the latest news about the selected stock from Google News RSS.

---

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/):** For building the interactive user interface.
- **[yfinance](https://pypi.org/project/yfinance/):** To fetch real-time and historical stock data.
- **[Plotly](https://plotly.com/python/):** For advanced interactive visualizations.
- **[Backtrader](https://www.backtrader.com/):** For implementing and testing trading strategies.
- **[feedparser](https://pypi.org/project/feedparser/):** To parse RSS news feeds.

---

## 🧠 Educational Purpose

This project was created to help students:
- Understand how to combine Python libraries to create meaningful, interactive data applications.
- Learn the basics of technical indicators like Bollinger Bands and SMA.
- Explore the logic behind simple trading strategies and evaluate them through metrics.
- See the impact of factors like inflation on investment performance.

---

## 📸 Sample Screenshots

- 📊 Stock chart with Bollinger Bands and SMA
- 🧮 Backtest results: final capital, return %, Sharpe ratio, and more
- 📰 Latest news related to the selected stock

---

## 🏁 How to Run

the app is delpoyed: https://aramosdemostocks.streamlit.app/

But in case it is down (as I only got the free account), you can create an environment and:

```bash
pip install -r requirements.txt
streamlit run app.py
```
