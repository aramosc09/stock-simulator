import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import feedparser
import backtrader as bt

st.set_page_config(page_title="Streamlit Demo ARAMOSC", layout="wide")

# Initialize session state for running main logic
if "run_main" not in st.session_state:
    st.session_state.run_main = False

st.title("STOCK VISUALIZER")

with st.sidebar:
    st.header("Customize")
    ticker = st.text_input("Enter Ticker (Ex: AAPL, TSLA, MSFT)", value="AAPL")
    period = st.selectbox(
        "Select period",
        options=["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
        index=5
    )
    if st.button("Go", key="search_input"):
        st.session_state.run_main = True

if st.session_state.run_main:
    try:
        info = yf.Ticker(ticker.upper()).info
        company_name = info.get("shortName", ticker.upper())
        stock_data = yf.download(ticker.upper(), period=period, group_by='ticker', progress=False)
        stock_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        stock_data.index.name = 'Date'
        stock_data = stock_data.dropna()
        if not stock_data.empty:
            stock_data.reset_index(inplace=True)
            window = 20
            stock_data['SMA'] = stock_data['Close'].rolling(window=window).mean()
            stock_data['Upper Band'] = stock_data['SMA'] + 2 * stock_data['Close'].rolling(window=window).std()
            stock_data['Lower Band'] = stock_data['SMA'] - 2 * stock_data['Close'].rolling(window=window).std()

            st.subheader(f"{company_name} | {ticker.upper()} (DF showing up to last {10})")
            st.dataframe(stock_data.tail(10))

            csv = stock_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f'{ticker.upper()}_data.csv',
                mime='text/csv'
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name='Close'))

            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Upper Band'], name='Upper Band', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Lower Band'], name='Lower Band', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['SMA'], name='SMA', line=dict(dash='dash')))

            fig.update_layout(title=f"Stock close price {company_name} | {ticker.upper()}", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            tab1, tab2, = st.tabs(["Strategies", "News"])

            with tab1:
                st.header("Design your Strategy")
                st.subheader("Parameters")
                capital_inicial = st.number_input("Initial (USD)", min_value=1000, max_value=500000, value=1000, step=1000)
                period_bt = st.selectbox("Backtest period", options=["6mo", "1y", "2y", "3y", "5y"], index=4)
                inflacion_anual = st.number_input("Est. annual inflation(%)", min_value=0.0, value=4.0, step=0.5)
                agresiveness = st.slider("Agresiveness Level (Buy)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
                panic = st.slider("Panic Level (Sell)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
                if st.button("Go", key="strategies_button"):
                    class MyStrategy(bt.Strategy):
                        params = (
                            ("agresiveness", agresiveness),
                            ("panic", panic),
                        )

                        def __init__(self):
                            self.order = None
                            self.bb = bt.indicators.BollingerBands(self.data.close, period=20)
                            self.executions = []

                        def next(self):
                            if self.order:
                                return

                            price = self.data.close[0]

                            if price < self.bb.lines.bot[0]:
                                cash = self.broker.get_cash()
                                size = int((cash * self.params.agresiveness) / price)
                                if size > 0:
                                    self.order = self.buy(size=size)

                            elif self.position and price > self.bb.lines.top[0]:
                                sell_size = int(self.position.size * self.params.panic)
                                if sell_size > 0:
                                    self.order = self.sell(size=sell_size)

                        def notify_order(self, order):
                            if order.status in [order.Completed]:
                                exec_date = bt.num2date(order.executed.dt).date()
                                self.executions.append({
                                    "date": exec_date,
                                    "price": order.executed.price,
                                    "size": order.executed.size
                                })
                            self.order = None

                    class TradeStats(bt.Analyzer):
                        def __init__(self):
                            self.trades = []

                        def notify_trade(self, trade):
                            if trade.isclosed:
                                print(f"TRADE CLOSED: pnl={trade.pnl}, dt={trade.close_datetime()}")
                                self.trades.append(trade.pnl)

                    class ValueTracker(bt.Analyzer):
                        def __init__(self):
                            self.values = []

                        def next(self):
                            self.values.append(self.strategy.broker.get_value())

                    # CashTracker Analyzer
                    class CashTracker(bt.Analyzer):
                        def __init__(self):
                            self.cash = []

                        def next(self):
                            self.cash.append(self.strategy.broker.get_cash())

                    df_bt = yf.download(ticker, period=period_bt, group_by='ticker', progress=False)
                    df_bt.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    df_bt.index.name = 'datetime'
                    df_bt = df_bt.dropna()

                    if len(df_bt) > 30:
                        cerebro = bt.Cerebro()
                        cerebro.broker.setcash(capital_inicial)
                        cerebro.addstrategy(MyStrategy, agresiveness=agresiveness, panic=panic)
                        data = bt.feeds.PandasData(dataname=df_bt)
                        cerebro.adddata(data)
                        cerebro.addanalyzer(TradeStats, _name='stats')
                        cerebro.addanalyzer(ValueTracker, _name='valtrack')
                        cerebro.addanalyzer(CashTracker, _name='cash')
                        results = cerebro.run()
                        stats = results[0].analyzers.stats.trades
                        values = results[0].analyzers.valtrack.values
                        cash_values = results[0].analyzers.cash.cash

                        initial_cash = capital_inicial
                        final_value = cerebro.broker.get_value()
                        profit = final_value - initial_cash
                        profit_pct = (profit / initial_cash) * 100
                        period_years = {"6mo": 0.5, "1y": 1, "2y": 2, "3y": 3, "5y": 5}[period_bt]
                        estimated_inflation_pct = inflacion_anual * period_years
                        adjusted_return = profit_pct - estimated_inflation_pct

                        st.header("📊 Strategy Results")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Initial $", f"${initial_cash:,.2f}")
                        col2.metric("Final $", f"${final_value:,.2f}")
                        col3.metric("Win/Loss (before inflation adjustment)", f"${profit:,.2f}")

                        col4, col5, col6 = st.columns(3)
                        col4.metric("Rentability", f"{profit_pct:.2f}%")
                        col5.metric("Adjusted rentability", f"{adjusted_return:.2f}%")
                        col6.metric("Est. Inflation", f"{estimated_inflation_pct:.2f}%")

                        #STATS SAVING
                        total_trades = 0
                        winning_trades = 0
                        losing_trades = 0
                        gains = []
                        losses = []

                        positions = []
                        executions = results[0].executions

                        for exec in executions:
                            date = exec["date"]
                            price = exec["price"]
                            size = exec["size"]
                            if size > 0:
                                positions.append({"price": price, "size": size})
                            elif size < 0:
                                size_to_sell = abs(size)
                                realized_pnl = 0
                                while size_to_sell > 0 and positions:
                                    buy = positions.pop(0)
                                    qty_matched = min(size_to_sell, buy["size"])
                                    pnl = (price - buy["price"]) * qty_matched
                                    realized_pnl += pnl
                                    if qty_matched < buy["size"]:
                                        positions.insert(0, {"price": buy["price"], "size": buy["size"] - qty_matched})
                                    size_to_sell -= qty_matched
                                if realized_pnl > 0:
                                    winning_trades += 1
                                    gains.append(realized_pnl)
                                elif realized_pnl < 0:
                                    losing_trades += 1
                                    losses.append(abs(realized_pnl))
                                total_trades += 1

                        win_rate = (winning_trades / total_trades * 100) if total_trades else 0
                        total_gain = sum(gains)
                        total_loss = sum(losses)
                        profit_factor = (total_gain / total_loss) if total_loss != 0 else float('inf')
                        expectancy = ((winning_trades / total_trades) * (total_gain / winning_trades) if winning_trades > 0 else 0) + \
                                     ((losing_trades / total_trades) * (-total_loss / losing_trades) if losing_trades > 0 else 0)

                        max_drawdown = 0
                        peak = values[0]
                        for value in values:
                            if value > peak:
                                peak = value
                            dd = (peak - value) / peak
                            max_drawdown = max(max_drawdown, dd)

                        returns = [(values[i+1] - values[i]) / values[i] for i in range(len(values) - 1)]
                        if len(returns) > 1:
                            avg_return = sum(returns) / len(returns)
                            std_return = (sum((r - avg_return)**2 for r in returns) / (len(returns) - 1))**0.5
                            sharpe_ratio = avg_return / std_return if std_return != 0 else float('inf')
                        else:
                            sharpe_ratio = 0

                        operations_col, metrics_col = st.columns(2)

                        with operations_col:
                            with st.expander("Operations"):
                                st.write(f"**Total operations:** {total_trades}")
                                st.write(f"**Wins:** {winning_trades}")
                                st.write(f"**Losses:** {losing_trades}")
                                st.write(f"**Win rate:** {win_rate:.2f}%")
                        with metrics_col:
                            with st.expander("Advanced metrics"):
                                st.write(f"**Max Drawdown:** {max_drawdown * 100:.2f}%")
                                st.write(f"**Sharpe Ratio:** {sharpe_ratio:.2f}")
                                st.write(f"**Profit Factor:** {profit_factor:.2f}" if profit_factor != float('inf') else "**Profit Factor:** N/A (no losses)")
                                st.write(f"**Expectancy:** {expectancy:.2f}")

                        # Portfolio Evolution Graph
                        dates = df_bt.index[-len(values):]
                        invested_values = [v - c for v, c in zip(values, cash_values)]

                        fig_perf = go.Figure()
                        fig_perf.add_trace(go.Scatter(x=dates, y=values, mode='lines', name='Total Value', line=dict(color='blue')))
                        fig_perf.add_trace(go.Scatter(x=dates, y=invested_values, mode='lines', name='Invested', line=dict(color='green')))
                        fig_perf.add_trace(go.Scatter(x=dates, y=cash_values, mode='lines', name='Cash', line=dict(color='orange')))

                        fig_perf.update_layout(title="Portfolio Evolution", xaxis_title="Date", yaxis_title="Value (USD)", height=500)
                        # Identify new highs and lows
                        new_highs = []
                        new_lows = []
                        last_high = values[0]
                        last_low = values[0]
                        for i, v in enumerate(values):
                            if v > last_high:
                                new_highs.append((dates[i], v))
                                last_high = v
                            if v < last_low:
                                new_lows.append((dates[i], v))
                                last_low = v

                        # Add markers for new highs
                        if new_highs:
                            fig_perf.add_trace(go.Scatter(
                                x=[d[0] for d in new_highs],
                                y=[d[1] for d in new_highs],
                                mode='markers',
                                name='New High',
                                marker=dict(color='lime', size=10, symbol='triangle-up'),
                                hovertext=[f"New High: ${v:.2f}" for (_, v) in new_highs],
                                hoverinfo='text'
                            ))

                        # Add markers for new lows
                        if new_lows:
                            fig_perf.add_trace(go.Scatter(
                                x=[d[0] for d in new_lows],
                                y=[d[1] for d in new_lows],
                                mode='markers',
                                name='New Low',
                                marker=dict(color='red', size=10, symbol='triangle-down'),
                                hovertext=[f"New Low: ${v:.2f}" for (_, v) in new_lows],
                                hoverinfo='text'
                            ))

                        # Update chart again with highlights
                        fig_perf.update_layout(showlegend=True)
                        st.plotly_chart(fig_perf, use_container_width=True)

                        # GRAPH: Agrupar ejecuciones reales desde notify_order
                        executions = results[0].executions

                        from collections import defaultdict
                        buy_summary = defaultdict(lambda: {"total_qty": 0, "total_val": 0})
                        sell_summary = defaultdict(lambda: {"total_qty": 0, "total_val": 0})

                        for exec in executions:
                            date = exec["date"]
                            price = exec["price"]
                            size = exec["size"]
                            if size > 0:
                                buy_summary[date]["total_qty"] += size
                                buy_summary[date]["total_val"] += price * size
                            elif size < 0:
                                sell_summary[date]["total_qty"] += abs(size)
                                sell_summary[date]["total_val"] += abs(size) * price

                        buy_dates = []
                        buy_prices = []
                        buy_hover = []
                        sell_dates = []
                        sell_prices = []
                        sell_hover = []

                        for date, stats in buy_summary.items():
                            avg_price = stats["total_val"] / stats["total_qty"]
                            close_price = df_bt.loc[str(date), "Close"] if str(date) in df_bt.index.astype(str) else avg_price
                            buy_dates.append(date)
                            buy_prices.append(avg_price)
                            buy_hover.append(f"Date: {date}<br>Avg Trade Price: ${avg_price:.2f}<br>Close: ${close_price:.2f}<br>Qty: {stats['total_qty']}")

                        for date, stats in sell_summary.items():
                            avg_price = stats["total_val"] / stats["total_qty"]
                            close_price = df_bt.loc[str(date), "Close"] if str(date) in df_bt.index.astype(str) else avg_price
                            sell_dates.append(date)
                            sell_prices.append(avg_price)
                            sell_hover.append(f"Date: {date}<br>Avg Trade Price: ${avg_price:.2f}<br>Close: ${close_price:.2f}<br>Qty: {stats['total_qty']}")

                        #ARROWS
                        print("INIT GRAPH")
                        # Calculate SMA and Bollinger Bands if not already present
                        if 'SMA' not in df_bt.columns or 'Upper Band' not in df_bt.columns or 'Lower Band' not in df_bt.columns:
                            df_bt['SMA'] = df_bt['Close'].rolling(window=20).mean()
                            df_bt['Upper Band'] = df_bt['SMA'] + 2 * df_bt['Close'].rolling(window=20).std()
                            df_bt['Lower Band'] = df_bt['SMA'] - 2 * df_bt['Close'].rolling(window=20).std()

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Close'], name='Close', line=dict(color='blue')))
                        # Add Bollinger Bands and SMA traces
                        fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Upper Band'], name='Upper Band', line=dict(dash='dot')))
                        fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Lower Band'], name='Lower Band', line=dict(dash='dot')))
                        fig.add_trace(go.Scatter(x=df_bt.index, y=df_bt['SMA'], name='SMA', line=dict(dash='dash')))
                        print("FIRST TRACE")

                        fig.add_trace(go.Scatter(
                            x=buy_dates,
                            y=buy_prices,
                            mode="markers",
                            name="Buy",
                            marker=dict(color="green", size=10),
                            hovertext=buy_hover,
                            hoverinfo="text"
                        ))
                        print("SECOND TRACE")

                        fig.add_trace(go.Scatter(
                            x=sell_dates,
                            y=sell_prices,
                            mode="markers",
                            name="Sell",
                            marker=dict(color="red", size=10),
                            hovertext=sell_hover,
                            hoverinfo="text"
                        ))
                        print("THIRD TRACE")

                        fig.update_layout(
                            title="Strategy Operation Grapgh",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            height=500,
                            xaxis=dict(tickformat="%Y-%m-%d")
                        )
                        print("UPDATE")
                        st.plotly_chart(fig, use_container_width=True)
                        print("UPDATE ST")
                    else:
                        st.warning("Not wnough data to run strategy. T-T")

            with tab2:
                st.header(f"{company_name} | {ticker.upper()} News")
                clean_ticker = ticker.split('.')[0]
                country = st.selectbox("Select News Region", options=["USA", "MX"], index=0)
                lang_gl_ceid = {
                    "USA": "en-US&gl=US&ceid=US:en",
                    "MX": "es-419&gl=MX&ceid=MX:es-419"
                }
                url = f"https://news.google.com/rss/search?q={ticker}+stock&hl={lang_gl_ceid[country]}"

                # url = f"https://news.google.com/rss/search?q={clean_ticker}+stock&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(url)

                if not feed.entries:
                    st.warning("No se encontraron noticias recientes.")
                else:
                    for entry in feed.entries[:5]:
                        st.subheader(entry.title)
                        st.write(entry.published)
                        st.markdown(f"[More Info]({entry.link})")
                        st.markdown("---")

        else:
            st.warning("Unable to find data. Check ticker input.")

    except Exception as e:
        st.error(f"Error: {e}")