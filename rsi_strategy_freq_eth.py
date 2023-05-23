

# ----------------- 
# Load libraries 
# -----------------

import pandas as pd
import numpy as np
import datetime

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default = "browser"


# -----------------
# Define functions
# -----------------

#  ---------- define rsi ---------- #

def rsi(price,n=14):
    delta = price['close'].diff()
    dUp,dDown = delta.copy(), delta.copy()
    dUp[dUp<0] = 0
    dDown[dDown>0] = 0

    RolUp = dUp.rolling(window = n).mean()
    RolDown = dDown.rolling(window = n).mean().abs()

    RS = RolUp/RolDown
    rsi = 100.0 - (100.0/(1.0+RS))

    return rsi 


# ------------------------------------
# Define last local minima functions
# ------------------------------------

def find_local_minima_last(df, current_candle_pos, RANGE_CHECK_LOCAL_MINIMA, LOCAL_DEEP_VAR):
    start_pos = current_candle_pos - RANGE_CHECK_LOCAL_MINIMA
    end_pos = current_candle_pos + 1
    previous_candles = df['RSI'].iloc[start_pos:end_pos]
    local_minima_positions = previous_candles[(previous_candles <= previous_candles.shift(LOCAL_DEEP_VAR)) &
                                              (previous_candles <= previous_candles.shift(-LOCAL_DEEP_VAR))].index.tolist()
    if len(local_minima_positions) > 0:
        local_minima_pos_last = local_minima_positions[-1]
        return local_minima_pos_last
    else:
        return -1
    
def find_local_minima_first(df, current_candle_pos,WINDOW_RANGE, RANGE_CHECK_LOCAL_MINIMA, LOCAL_DEEP_VAR):
    start_pos = current_candle_pos - WINDOW_RANGE - RANGE_CHECK_LOCAL_MINIMA
    end_pos = current_candle_pos - RANGE_CHECK_LOCAL_MINIMA + 1
    previous_candles = df['RSI'].iloc[start_pos:end_pos]
    local_minima_positions = previous_candles[(previous_candles <= previous_candles.shift(LOCAL_DEEP_VAR)) &
                                              (previous_candles <= previous_candles.shift(-LOCAL_DEEP_VAR))].index.tolist()
    if len(local_minima_positions) > 0:
        local_minima_pos_first = local_minima_positions[0]
        return local_minima_pos_first
    else:
        return -1


# --------------------------------
# Plot functions 
# --------------------------------

def plot_candle_rsi(df, candle_pos):
    # Get init position for the plot 
    local_minima_pos_first = int(df.loc[candle_pos, 'local_minima_pos_first'])
    local_minima_pos_last = int(df.loc[candle_pos, 'local_minima_pos_last'])
    start_pos = int(local_minima_pos_first - 5)
    end_pos = candle_pos + 10

    # Create a sub dataframe for the plot range
    df_plot = df.iloc[start_pos:end_pos]

    # Create subplot figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=('Price', 'RSI'), 
                        vertical_spacing=0.1)

    # Add candlestick trace to the subplot figure (row 1)
    fig.add_trace(go.Candlestick(x=df_plot.index,
                    open=df_plot['open'],
                    high=df_plot['high'],
                    low=df_plot['low'],
                    close=df_plot['close'],
                    name = "Candlesticks"), row=1, col=1)

    # Add RSI trace to the subplot figure (row 2)
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], mode='lines', name='RSI', line=dict(color='blue')), row=2, col=1)

    # Add local minima points for RSI
    fig.add_trace(go.Scatter(
        x=[df_plot.index[local_minima_pos_first - start_pos], df_plot.index[local_minima_pos_last - start_pos]],
        y=[df_plot.loc[local_minima_pos_first, 'RSI'], df_plot.loc[local_minima_pos_last, 'RSI']],
        mode='markers',
        name='Local Minima RSI',
        marker=dict(
            size=4,
            color='yellow',
        )
    ), row=2, col=1)

    # Add local minima points for Price
    fig.add_trace(go.Scatter(
        x=[df_plot.index[local_minima_pos_first - start_pos], df_plot.index[local_minima_pos_last - start_pos]],
        y=[df_plot.loc[local_minima_pos_first, 'low'] - 0.5, df_plot.loc[local_minima_pos_last, 'low'] - 0.5], # Subtract a small value from the low price at local minima positions
        mode='markers',
        name='Local Minima Price',
        marker=dict(
            size=4,
            color='yellow',
        )
    ), row=1, col=1)

    # Add lines connecting the local minima
    fig.add_trace(go.Scatter(
        x=[df_plot.index[local_minima_pos_first - start_pos], df_plot.index[local_minima_pos_last - start_pos]],
        y=[df_plot.loc[local_minima_pos_first, 'RSI'], df_plot.loc[local_minima_pos_last, 'RSI']],
        mode='lines',
        name='Local Minima Line RSI',
        line=dict(color='yellow', width=0.5)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[df_plot.index[local_minima_pos_first - start_pos], df_plot.index[local_minima_pos_last - start_pos]],
        y=[df_plot.loc[local_minima_pos_first, 'low'], df_plot.loc[local_minima_pos_last, 'low']],
        mode='lines',
        name='Local Minima Line Price',
        line=dict(color='yellow', width=0.5)
    ), row=1, col=1)

    # Mark the candle_pos with 'X'
    fig.add_trace(go.Scatter(
        x=[df_plot.index[candle_pos - start_pos]],
        y=[df_plot.loc[candle_pos, 'low']],
        mode='markers',
        name='Current Position',
        marker=dict(
            size=10,
            color='orange',
            symbol='x'
        )
    ), row=1, col=1)

    # Set plot layout to dark and adjust to full width
    fig.update_layout(height=600, width=1200, title_text="Divergence Detection", template='plotly_dark')
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=False)))
    fig.show()


# ----------------------
# Initalize parameters
# ----------------------

TOKEN = 'eth'

START_YEAR = 2021
START_MONTH = 1
START_DAY = 1
LOOKBACK = 60

#start_date = '2021-01-18 12:02:00'
#end_date = '2022-01-21 12:02:00'
FREQUENCY = '5T'

WINDOW_RANGE = 30

STARTING_CAPITAL = 100 

RSI_LOOKBACK = 14

RSI_OVERSOLD = 40

PLOT_POSITION = False 

RANGE_CHECK_LOCAL_MINIMA = 20
LOCAL_DEEP_VAR = 5

# Assuming you have a column called 'trailing_stop_loss' representing the trailing stop loss level
TRAILING_STOP_LOSS_LEVEL = -0.2  # Define your desired trailing stop loss level

# Initialize max_return and stop_loss_level
max_return = 0
stop_loss_level = 0
available_capital = 100  # Total available capital to invest
position_size = 10  # Desired position size per trade

TRADING_FEE = 0.001 

RISK_FRACTION = 0.2  

# ------------------
# Get data
# ------------------

file_name = f'400pais_1m/{TOKEN}usd'

# Load data 
df_hist = pd.read_csv('./data/'+file_name+'.csv')


# ------------------
# Prepare data 
# ------------------

# date to datetime
df_hist['date'] = pd.to_datetime(df_hist['time'], unit='ms')

# calculate price column from high and low
df_hist['price'] = (df_hist.high + df_hist.low)/2

# reorder columns time price 
df_hist = df_hist[['date', 'price']]

# resample data to desired frequency and get the mean, max, min, open, close
df_resampled = df_hist.resample(FREQUENCY, on='date').agg({'price':  ['mean', 'max', 'min', 'first', 'last']})

# remove NA 
df_resampled = df_resampled.dropna()

df_resampled.columns = df_resampled.columns.droplevel()

# column date from index 
df_resampled['date'] = df_resampled.index

# reset index
df_resampled = df_resampled.reset_index(drop=True)

# rename columns 
df_resampled = df_resampled.rename(columns={'first': 'open', 'last': 'close', 'max': 'high', 'min': 'low'})

# reorder columns
df_resampled = df_resampled[['date', 'open', 'high', 'low', 'close', 'mean']]


# ----------------------------
# Prepare data for backtest 
# ----------------------------

df_hist = df_resampled.copy()

del df_resampled

# Filter to start date 
df_hist = df_hist[df_hist.date >= datetime.datetime(START_YEAR,START_MONTH,START_DAY)]

# Reset index
df_hist = df_hist.reset_index(drop=True)


# -------------------- 
# Get Signal
# --------------------

df = df_hist.copy()
df = df.reset_index(drop=True)
df['RSI'] = rsi(df,n=RSI_LOOKBACK)


POS_RANGE = range(50,len(df)-1) # 730826)

for pos in POS_RANGE:

    current_candle_pos = pos
    
    local_minima_pos_last = find_local_minima_last(df, current_candle_pos, RANGE_CHECK_LOCAL_MINIMA, LOCAL_DEEP_VAR)
    local_minima_pos_first = find_local_minima_first(df, current_candle_pos, WINDOW_RANGE, RANGE_CHECK_LOCAL_MINIMA, LOCAL_DEEP_VAR)
    
    if local_minima_pos_last >0 and local_minima_pos_first >0:

        df.loc[pos, 'local_minima_pos_first'] = local_minima_pos_first
        df.loc[pos, 'local_minima_pos_last'] = local_minima_pos_last

        df.loc[pos, 'local_minima_price_first'] = df.loc[local_minima_pos_first, 'mean']
        df.loc[pos, 'local_minima_price_last'] = df.loc[local_minima_pos_last, 'mean']

        df.loc[pos, 'local_minima_rsi_first'] = df.loc[local_minima_pos_first, 'RSI']
        df.loc[pos, 'local_minima_rsi_last'] = df.loc[local_minima_pos_last, 'RSI']


# Create slope between local minima and local maxima
df['slope_price'] = (df['local_minima_price_last'] - df['local_minima_price_first']) / (
            df['local_minima_pos_last'] - df['local_minima_pos_first'])

df['slope_rsi'] = (df['local_minima_rsi_last'] - df['local_minima_rsi_first']) / (
            df['local_minima_pos_last'] - df['local_minima_pos_first'])


# Create descending divergence signal
RSI_OVERSOLD = 40
df['Signal_Divergence'] = 0
df.loc[(df['slope_price'] < 0) 
       & (df['slope_rsi'] > 0)
       & (df['RSI'] < RSI_OVERSOLD)
       & (df['local_minima_rsi_first'] < RSI_OVERSOLD)
       & (df['local_minima_rsi_last'] < RSI_OVERSOLD)
       , 'Signal_Divergence'] = 1

# Get candle position where the signal is 1 
candle_pos_list = df[df['Signal_Divergence'] == 1].index.tolist()

print('We have {} signals'.format(len(candle_pos_list)))


# -----------------------
# Plot candle position
# -----------------------

if PLOT_POSITION:
    # Get init position for the plot 
    candle_pos = candle_pos_list[5]
    plot_candle_rsi(df, candle_pos_list[5])


# --------------------
# Backtest simple
# --------------------

# --------- Backtest simple ---------

df_range = df.loc[POS_RANGE]

initial_capital = 100  # Initial capital in dollars
investment_per_trade = 20  # Investment amount per trade in dollars
trading_fee = 0.001  # Assuming a trading fee of 1% per trade

# get price forward + 5 and price_forward_10 
df_range['price_forward_5'] = df_range['close'].shift(-5)
df_range['price_forward_10'] = df_range['close'].shift(-10)

# get the return for the next 5 and 10 candles
df_range['return_5'] = df_range['price_forward_5'] / df_range['close'] - 1
df_range['return_10'] = df_range['price_forward_10'] / df_range['close'] - 1

# Calculate the strategy returns without fees
df_range['strategy_return_without_fee_5'] = df_range['return_5'] * df_range['Signal_Divergence']
df_range['strategy_return_without_fee_10'] = df_range['return_10'] * df_range['Signal_Divergence']

# Calculate the strategy returns with fees
df_range['strategy_return_with_fee_5'] = df_range['strategy_return_without_fee_5'] * (1 - df_range['Signal_Divergence'] * trading_fee)
df_range['strategy_return_with_fee_10'] = df_range['strategy_return_without_fee_10'] * (1 - df_range['Signal_Divergence'] * trading_fee)

# Calculate the cumulative return of the strategy without fees
df_range['cum_strategy_return_without_fee_5'] = (df_range['strategy_return_without_fee_5'] + 1).cumprod()
df_range['cum_strategy_return_without_fee_10'] = (df_range['strategy_return_without_fee_10'] + 1).cumprod()

# Calculate the cumulative return of the strategy with fees
df_range['cum_strategy_return_with_fee_5'] = (df_range['strategy_return_with_fee_5'] + 1).cumprod()
df_range['cum_strategy_return_with_fee_10'] = (df_range['strategy_return_with_fee_10'] + 1).cumprod()

# Multiply by the initial capital and investment per trade
df_range['cum_strategy_return_without_fee_5'] = df_range['cum_strategy_return_without_fee_5'] * initial_capital 
df_range['cum_strategy_return_without_fee_10'] = df_range['cum_strategy_return_without_fee_10'] * initial_capital 
df_range['cum_strategy_return_with_fee_5'] = df_range['cum_strategy_return_with_fee_5'] * initial_capital 
df_range['cum_strategy_return_with_fee_10'] = df_range['cum_strategy_return_with_fee_10'] * initial_capital 

# Calculate the cumulative return of the hold out return
df_range['cum_hold_out_return_5'] = (df_range['return_5'] + 1).cumprod() * initial_capital 
df_range['cum_hold_out_return_10'] = (df_range['return_10'] + 1).cumprod() * initial_capital 


#  ----------- Plot Backtest simple ----------------

# Plot
fig = make_subplots(rows=1, cols=1)

# Add cumulative return of the strategy with fees and position sizing
fig.add_trace(
    go.Scatter(x=df_range.date, y=df_range['cum_strategy_return_with_fee_5'], name='Strategy Return with Fee 5'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df_range.date, y=df_range['cum_strategy_return_with_fee_10'], name='Strategy Return with Fee 10'),
    row=1, col=1
)

# Add cumulative return of the strategy without fees and position sizing
fig.add_trace(
    go.Scatter(x=df_range.date, y=df_range['cum_strategy_return_without_fee_5'], name='Strategy Return without Fee 5'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df_range.date, y=df_range['cum_strategy_return_without_fee_10'], name='Strategy Return without Fee 10'),
    row=1, col=1
)

# Add cumulative return of the hold out return with position sizing
fig.add_trace(
    go.Scatter(x=df_range.date, y=df_range['cum_hold_out_return_5'], name='Hold Out Return 5'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df_range.date, y=df_range['cum_hold_out_return_10'], name='Hold Out Return 10'),
    row=1, col=1
)

# Update plot layout
fig.update_layout(
    title='Backtest Cumulative Returns',
    xaxis_title='Time',
    yaxis_title='Cumulative Return',
    template='plotly_dark'
)

fig.show()


# ----------- Backtest Analysis ------------

# Calculate the strategy returns with fees
df_range['strategy_return'] = df_range['strategy_return_with_fee_5']
df_range['cum_strategy_return'] = (df_range['strategy_return'] + 1).cumprod()

# Profit factor calculation
gross_profit = df_range[df_range['strategy_return'] > 0]['strategy_return'].sum()
gross_loss = abs(df_range[df_range['strategy_return'] < 0]['strategy_return'].sum())
profit_factor = gross_profit / gross_loss

# Sharpe ratio calculation
# Here, I'll assume a risk-free rate of 0. The risk-free rate can be changed based on your specific context.
risk_free_rate = 0
excess_daily_returns = df_range['strategy_return'] - risk_free_rate
sharpe_ratio = np.sqrt(len(df_range)) * excess_daily_returns.mean() / excess_daily_returns.std()

# Calculate number of winning and losing trades
num_winning_trades = (df_range['strategy_return'] > 0).sum()
num_losing_trades = (df_range['strategy_return'] < 0).sum()

# Print the results
print(f'Profit Factor: {profit_factor}')
print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Number of Winning Trades: {num_winning_trades}')
print(f'Number of Losing Trades: {num_losing_trades}')


# -------------------------
# Backtest with stop loss
# -------------------------

STOP_LOSS = 0.01  # % stop loss

# --------- Backtest stop loss  ---------

df_range = df.loc[POS_RANGE]

initial_capital = 100  # Initial capital in dollars
trading_fee = 0.001  # Assuming a trading fee of 1% per trade

# Get price forward + 5 and price_forward_10 
df_range['price_forward_5'] = df_range['close'].shift(-5)
df_range['price_forward_10'] = df_range['close'].shift(-10)

# Get the return for the next 5 and 10 candles
df_range['return_5'] = df_range['price_forward_5'] / df_range['close'] - 1
df_range['return_10'] = df_range['price_forward_10'] / df_range['close'] - 1

# Calculate the min price for the next 5 and 10 candles
df_range['min_price_next_5'] = df_range['close'].rolling(window=5).min().shift(-5)
df_range['min_price_next_10'] = df_range['close'].rolling(window=10).min().shift(-10)

# Determine if stop loss is hit
df_range['stop_loss_hit_5'] = df_range['min_price_next_5'] < df_range['close'] * (1 - STOP_LOSS)
df_range['stop_loss_hit_10'] = df_range['min_price_next_10'] < df_range['close'] * (1 - STOP_LOSS)

# Adjust returns if stop loss is hit
df_range.loc[df_range['stop_loss_hit_5'], 'return_5'] = -STOP_LOSS
df_range.loc[df_range['stop_loss_hit_10'], 'return_10'] = -STOP_LOSS

# Calculate the strategy returns without fees
df_range['strategy_return_without_fee_5'] = df_range['return_5'] * df_range['Signal_Divergence']
df_range['strategy_return_without_fee_10'] = df_range['return_10'] * df_range['Signal_Divergence']

# Calculate the strategy returns with fees
df_range['strategy_return_with_fee_5'] = df_range['strategy_return_without_fee_5'] * (1 - df_range['Signal_Divergence'] * trading_fee)
df_range['strategy_return_with_fee_10'] = df_range['strategy_return_without_fee_10'] * (1 - df_range['Signal_Divergence'] * trading_fee)

# Calculate the cumulative return of the strategy without fees
df_range['cum_strategy_return_without_fee_5'] = (df_range['strategy_return_without_fee_5'] + 1).cumprod()
df_range['cum_strategy_return_without_fee_10'] = (df_range['strategy_return_without_fee_10'] + 1).cumprod()

# Calculate the cumulative return of the strategy with fees
df_range['cum_strategy_return_with_fee_5'] = (df_range['strategy_return_with_fee_5'] + 1).cumprod()
df_range['cum_strategy_return_with_fee_10'] = (df_range['strategy_return_with_fee_10'] + 1).cumprod()

# Multiply by the initial capital
df_range['cum_strategy_return_without_fee_5'] = df_range['cum_strategy_return_without_fee_5'] * initial_capital 
df_range['cum_strategy_return_without_fee_10'] = df_range['cum_strategy_return_without_fee_10'] * initial_capital 
df_range['cum_strategy_return_with_fee_5'] = df_range['cum_strategy_return_with_fee_5'] * initial_capital 
df_range['cum_strategy_return_with_fee_10'] = df_range['cum_strategy_return_with_fee_10'] * initial_capital 

# Calculate the cumulative return of the hold out return
df_range['cum_hold_out_return_5'] = (df_range['return_5'] + 1).cumprod() * initial_capital 
df_range['cum_hold_out_return_10'] = (df_range['return_10'] + 1).cumprod() * initial_capital 


#  ----------- Plot Backtest simple ----------------

# Plot
fig = make_subplots(rows=1, cols=1)

# Add cumulative return of the strategy with fees and position sizing
fig.add_trace(
    go.Scatter(x=df_range.date, y=df_range['cum_strategy_return_with_fee_5'], name='Strategy Return with Fee 5'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df_range.date, y=df_range['cum_strategy_return_with_fee_10'], name='Strategy Return with Fee 10'),
    row=1, col=1
)

# Add cumulative return of the strategy without fees and position sizing
fig.add_trace(
    go.Scatter(x=df_range.date, y=df_range['cum_strategy_return_without_fee_5'], name='Strategy Return without Fee 5'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df_range.date, y=df_range['cum_strategy_return_without_fee_10'], name='Strategy Return without Fee 10'),
    row=1, col=1
)

# Add cumulative return of the hold out return with position sizing
fig.add_trace(
    go.Scatter(x=df_range.date, y=df_range['cum_hold_out_return_5'], name='Hold Out Return 5'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=df_range.date, y=df_range['cum_hold_out_return_10'], name='Hold Out Return 10'),
    row=1, col=1
)

# Update plot layout
fig.update_layout(
    title='Backtest Cumulative Returns with Stop loss',
    xaxis_title='Time',
    yaxis_title='Cumulative Return',
    template='plotly_dark'
)

fig.show()


# ----------- Backtest Analysis ------------

# Calculate the strategy returns with fees
df_range['strategy_return'] = df_range['strategy_return_with_fee_5']
df_range['cum_strategy_return'] = (df_range['strategy_return'] + 1).cumprod()

# Profit factor calculation
gross_profit = df_range[df_range['strategy_return'] > 0]['strategy_return'].sum()
gross_loss = abs(df_range[df_range['strategy_return'] < 0]['strategy_return'].sum())
profit_factor = gross_profit / gross_loss

# Sharpe ratio calculation
# Here, I'll assume a risk-free rate of 0. The risk-free rate can be changed based on your specific context.
risk_free_rate = 0
excess_daily_returns = df_range['strategy_return'] - risk_free_rate
sharpe_ratio = np.sqrt(len(df_range)) * excess_daily_returns.mean() / excess_daily_returns.std()

# Calculate number of winning and losing trades
num_winning_trades = (df_range['strategy_return'] > 0).sum()
num_losing_trades = (df_range['strategy_return'] < 0).sum()

# Print the results
print(f'Profit Factor: {profit_factor}')
print(f'Sharpe Ratio: {sharpe_ratio}')
print(f'Number of Winning Trades: {num_winning_trades}')
print(f'Number of Losing Trades: {num_losing_trades}')


# ----------------------------
# Save the backtest results
# ----------------------------

# Save the backtest results to pickle 
df_range.to_pickle(f'./backtest/backtest_results_{TOKEN}.pkl')

# save the plot 
fig.write_html(f'./backtest/backtest_plot_{TOKEN}.html')

