

# --------------------------------------
# Import libraries
# --------------------------------------

import websocket
import json
import time

import configparser
import requests

import pandas as pd 

from binance.client import Client

import numpy as np

import logging


# --------------------------------------
# Define binance client connection 
# --------------------------------------

config = configparser.ConfigParser()
config.read('config.ini')


api_key = config['binance']['key']
api_secret = config['binance']['secret']

client = Client(api_key, api_secret)


# --------------------------------------
# Define logger
# --------------------------------------

logger = logging.getLogger(__name__)

# save logger in log folder 
logging.basicConfig(filename='log/binance_websocket_test.log', 
                    level=logging.INFO, 
                    format='%(asctime)s %(message)s')

logger.addFilter(lambda record: not record.name.startswith('websockets'))


# ----------------------
# Initalize parameters
# ----------------------

global LOOKBACK
LOOKBACK = 60

global df

START_YEAR = 2021
START_MONTH = 1
START_DAY = 1
LOOKBACK = 60


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

global step_size
step_size = 0.0001

    # I want to buy 100 USD worth of BTC
global usdt_quantity
usdt_quantity = 20

global RUN_TYPE 
RUN_TYPE = 'TEST_TRADE' # 'TEST' or 'LIVE'

global open_trades
open_trades = []


# ----------------------
# Estrategy Functions 
# ----------------------

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

# -------- last local minima functions -------- #

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
    

# -------- Get signal -------- #

def get_signal(df):
    
    # Initialize signal column
    df['Signal_Divergence'] = 0
   
    # pos is the max index in data
    current_candle_pos = len(df) - 1
    
    local_minima_pos_last = find_local_minima_last(df, current_candle_pos, RANGE_CHECK_LOCAL_MINIMA, LOCAL_DEEP_VAR)
    local_minima_pos_first = find_local_minima_first(df, current_candle_pos, WINDOW_RANGE, RANGE_CHECK_LOCAL_MINIMA, LOCAL_DEEP_VAR)
    
    if local_minima_pos_last >0 and local_minima_pos_first >0:

        df.loc[current_candle_pos, 'local_minima_pos_first'] = local_minima_pos_first
        df.loc[current_candle_pos, 'local_minima_pos_last'] = local_minima_pos_last

        df.loc[current_candle_pos, 'local_minima_price_first'] = df.loc[local_minima_pos_first, 'mean']
        df.loc[current_candle_pos, 'local_minima_price_last'] = df.loc[local_minima_pos_last, 'mean']

        df.loc[current_candle_pos, 'local_minima_rsi_first'] = df.loc[local_minima_pos_first, 'RSI']
        df.loc[current_candle_pos, 'local_minima_rsi_last'] = df.loc[local_minima_pos_last, 'RSI']

        # Create slope between local minima and local maxima
        df['slope_price'] = (df['local_minima_price_last'] - df['local_minima_price_first']) / (
                    df['local_minima_pos_last'] - df['local_minima_pos_first'])

        df['slope_rsi'] = (df['local_minima_rsi_last'] - df['local_minima_rsi_first']) / (
                    df['local_minima_pos_last'] - df['local_minima_pos_first'])

        # Create descending divergence signal
        df.loc[(df['slope_price'] < 0) 
            & (df['slope_rsi'] > 0)
            & (df['RSI'] < RSI_OVERSOLD)
            & (df['local_minima_rsi_first'] < RSI_OVERSOLD)
            & (df['local_minima_rsi_last'] < RSI_OVERSOLD)
            , 'Signal_Divergence'] = 1
    
    # Get the signal
    Signal = False 
    if df['Signal_Divergence'].iloc[-1] == 1:
        Signal = True

    return Signal


# ------------------------
# Trading Functions
# ------------------------

def execute_buy_trade(df):

    if RUN_TYPE == 'TEST':
        print('***********Test mode, no trade executed')
        return

    # get last close price
    last_close_price = df['close'].iloc[-1]
    symbol = 'BTCUSDT'

    # Calculate the quantity of BTC to buy
    quantity = float(usdt_quantity / last_close_price)

    # Adjust the quantity to meet the step size requirement
    quantity = quantity - (quantity % step_size)

    try:
        buy_order = client.order_market_buy(symbol=symbol, quantity=quantity)
        print('***********Order executed')
        logging.info('Order executed' + str(buy_order))
        open_trades.append({'symbol': symbol, 'quantity': quantity, 'candle_counter': 0})
        print(buy_order)
    except Exception as e:
        print('***********Order failed')
        logging.info('Order failed' + str(e))
        print(e)


def execute_sell_order(trade):
    try:
        # Get balance before making the sell
        balance_before = client.get_asset_balance(asset='USDT')

        print(f"Balance before selling: {balance_before}")

        # Execute sell order
        sell_order = client.order_market_sell(
            symbol=trade['symbol'],
            quantity=trade['quantity']
        )
        print('***********Order executed')
        logging.info('Order executed' + str(sell_order))
        print(sell_order)

        # Get balance after making the sell
        balance_after = client.get_asset_balance(asset='USDT')

        print(f"Balance after selling: {balance_after}")

    except Exception as e:
        print(f"An error occurred - {e}")


def check_balance(symbol):
    balance = client.get_asset_balance(asset=symbol)
    return float(balance['free'])


# ------------------------
# Handle data Functions 
# ------------------------

def init_data():

    symbol = 'BTCUSDT'
    url = f'https://api.binance.com/api/v1/klines?symbol={symbol}&interval=5m&limit={LOOKBACK}'

    r = requests.get(url)

    data = r.json()

    import pandas as pd

    df = pd.DataFrame(data)

    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

    df.head()

    # transformt open_time to datetime object
    df['date'] = pd.to_datetime(df['open_time'], unit='ms')


    df = df[['date', 'open', 'high', 'low', 'close']]

    df.open = df.open.astype(float) # convert
    df.high = df.high.astype(float) # convert
    df.low = df.low.astype(float) # convert
    df.close = df.close.astype(float) # convert

    df['mean'] = (df.high + df.low)/2

    return df


def update_data(df, candle):

    df = pd.concat([df, pd.DataFrame(candle, index=[0])], ignore_index=True)

    if len(df)>LOOKBACK + 1:
        df = df.iloc[1:]

    df['RSI'] = rsi(df,n=RSI_LOOKBACK)

    return df


def signal_func(flag=[True]):
    # On the first call, flag[0] is True, so the function returns True
    # and then changes flag[0] to False
    # On subsequent calls, flag[0] is False, so the function returns False
    result = flag[0]
    flag[0] = False
    return result


# ----------------------
# Telegram functions 
# ----------------------



# ---------------------------------
# Performance tracking functions 
# ---------------------------------


# ---------------------
# Webshoket functions 
# ---------------------

def on_message(ws, message,df):

    data = json.loads(message)
    candlestick = data['k']

    if candlestick['x']:

        print('***********Candle completed. Analysis begins')
        
        # Get initial time before analysis 
        init_time_check = time.time()
        
        print('***********Get open trades')
        # Loop through all open trades
        for trade in open_trades:
            trade['candle_counter'] += 1

            if trade['candle_counter'] >= 5:
                print('Trade achieved 5 candles. Execute sell order')
                execute_sell_order(trade)

                # Remove the trade from the list of open trades
                open_trades.remove(trade)


        print('***********Get last candlestick')
        # Get candle info
        candlestick_data = {
            'date': pd.to_datetime(candlestick['t'], unit='ms'),
            'open': float(candlestick['o']),
            'high': float(candlestick['h']),
            'low': float(candlestick['l']),
            'close': float(candlestick['c'])
        }

        # Update data
        print('***********Update data')
        df = update_data(df, candlestick_data)

        # Calculate signal 
        print('***********Calculate signal')
        signal = get_signal(df)

        # Print signal
        print('***********Signal: ', signal)
        
        
        if RUN_TYPE == 'TEST_TRADE':
            signal = signal_func()

        print('***********Signal: ', signal)

        # Execute Trade 
        if signal:

            print('***********Execute Trade')

            # Get balance
            balance_before_trade = check_balance('USDT')
            print('***********Balance before trade: ', balance_before_trade)
            logging.info('Balance before trade:' + str(balance_before_trade))

            # Execute trade
            execute_buy_trade(df)

            # Update balance
            balance_after_trade = check_balance('USDT')
            print('***********Balance after trade: ', balance_after_trade)
            logging.info('Balance after trade:' + str(balance_after_trade))

       
        # Get final time after analysis
        final_time_check = time.time()
        
        # Get analysis duration in seconss 
        analysis_duration = final_time_check - init_time_check

        print('***********Analysis duration: ', analysis_duration)


def on_error(ws, error):
    print(error)  # Handle any errors that occur during the WebSocket connection
    logging.error(error)

def on_close(ws):
    print('***********WebSocket connection closed')  # Handle the WebSocket connection closing

def on_open(ws):
    # Subscribe to the 1-minute candlestick updates for BTC/USDT
    subscribe_data = {
        'method': 'SUBSCRIBE',
        'params': [
            'btcusdt@kline_5m'
        ],
        'id': 1
    }
    ws.send(json.dumps(subscribe_data))


if __name__ == '__main__':

    # Init data
    print('***********Initializing data')
    logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

    df = init_data()

    # Init client
    print('***********Initializing client')
    logging.info('Initializing client')

    client = Client(api_key, api_secret)

    # Init websocket
    print('***********Initializing websocket')
    logging.info('Initializing websocket')
    
    websocket.enableTrace(True)  # Enable WebSocket connection tracing (optional)
    ws = websocket.WebSocketApp(
        'wss://stream.binance.com:9443/ws',
        on_message=lambda ws, message: on_message(ws, message, df),
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = on_open

    ws.run_forever()
