

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

import datetime 

import csv

from telethon.sync import TelegramClient

from apscheduler.schedulers.background import BackgroundScheduler

from decimal import Decimal, getcontext



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

QUANTITY_PER_TRADE = 20 # Amount of dollars per transaction

RUN_TYPE = 'TEST_TRADE' # 'TEST' or 'LIVE'

open_trades = []

SYMBOL = 'BTCUSDT'

SYMBOL_webshocket = 'btcusdt'

UPDATE_FREQUENCY = 10 # 60 means 60 mnutues

TIMEFRAME = '1m' # Timeframe for the candles 5m = 5 minutes

LIMIT_OPEN_TRADES = 1 # Limit the number of open trades

global df 


# ----------------------
# Telegram functions 
# ----------------------

# connect telegram
phone = config['telegram']['phone']
api_id = config['telegram']['api_id']
api_hash = config['telegram']['api_hash']
api_messages = config['telegram']['api_messages']
chat_id_messages = config['telegram']['chat_id_messages']
api_alerts = config['telegram']['api_alerts']
chat_id_alerts = config['telegram']['chat_id_alerts']
bot_token = config['telegram']['token_rsi_bot']

# connect telegram
def connect_tg():

    client = TelegramClient(phone, api_id, api_hash)

    return client


def send_message_to_telegram(value):

     # message bot
    api_messages = '1823897212:AAG-arikVtpOO8zGZfkkoCKo9I1XzbYd2iA'
    chat_id_messages = str(556212849)

    # alerts bot
    api_alerts = '1683755311:AAFsOuP40Doy12JfTiFNzyIue2Fn_0XzWUg'
    chat_id_alerts = str(591016753)

    channel_api = 'bot'+ api_alerts
    chat_id = chat_id_alerts
    url = 'https://api.telegram.org/'+channel_api+'/sendMessage?chat_id=-'+chat_id+'&text="{}"'.format(value)
    requests.get(url)


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

    data = df.copy()
    
    # Initialize signal column
    data['Signal_Divergence'] = 0
   
    # pos is the max index in data
    current_candle_pos = len(data) - 1
    
    local_minima_pos_last = find_local_minima_last(data, current_candle_pos, RANGE_CHECK_LOCAL_MINIMA, LOCAL_DEEP_VAR)
    local_minima_pos_first = find_local_minima_first(data, current_candle_pos, WINDOW_RANGE, RANGE_CHECK_LOCAL_MINIMA, LOCAL_DEEP_VAR)
    
    if local_minima_pos_last >0 and local_minima_pos_first >0:

        data.loc[current_candle_pos, 'local_minima_pos_first'] = local_minima_pos_first
        data.loc[current_candle_pos, 'local_minima_pos_last'] = local_minima_pos_last

        data.loc[current_candle_pos, 'local_minima_price_first'] = data.loc[local_minima_pos_first, 'mean']
        data.loc[current_candle_pos, 'local_minima_price_last'] = data.loc[local_minima_pos_last, 'mean']

        data.loc[current_candle_pos, 'local_minima_rsi_first'] = data.loc[local_minima_pos_first, 'RSI']
        data.loc[current_candle_pos, 'local_minima_rsi_last'] = data.loc[local_minima_pos_last, 'RSI']

        # Create slope between local minima and local maxima
        data['slope_price'] = (data['local_minima_price_last'] - data['local_minima_price_first']) / (
                    data['local_minima_pos_last'] - data['local_minima_pos_first'])

        data['slope_rsi'] = (data['local_minima_rsi_last'] - data['local_minima_rsi_first']) / (
                    data['local_minima_pos_last'] - data['local_minima_pos_first'])

        # Create descending divergence signal
        data.loc[(data['slope_price'] < 0) 
            & (data['slope_rsi'] > 0)
            & (data['RSI'] < RSI_OVERSOLD)
            & (data['local_minima_rsi_first'] < RSI_OVERSOLD)
            & (data['local_minima_rsi_last'] < RSI_OVERSOLD)
            , 'Signal_Divergence'] = 1
    
    # Get the signal
    Signal = False 
    if data['Signal_Divergence'].iloc[-1] == 1:
        Signal = True

    return Signal


# ------------------------
# Trading Functions
# ------------------------

def get_step_size(SYMBOL):
    url = 'https://api.binance.com/api/v3/exchangeInfo'
    r = requests.get(url)
    exchange_info = r.json()

    # Find the symbol in the exchange info
    for s in exchange_info['symbols']:
        if s['symbol'] == SYMBOL:
            # Find the LOT_SIZE filter
            for filter in s['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    return float(filter['stepSize'])

    # If the symbol or LOT_SIZE filter was not found, return None
    return None

def quantiy_trade(df,QUANTITY_PER_TRADE,step_size):

    # Set precision
    getcontext().prec = 8

    # get last close price
    last_close_price = df['close'].iloc[-1]

    # Calculate the quantity of BTC to buy
    quantity = Decimal(QUANTITY_PER_TRADE / last_close_price)

    # Adjust the quantity to meet the step size requirement
    step_size = Decimal(step_size)
    quantity = quantity - (quantity % step_size)

    return float(quantity) 


def execute_buy_order(SYMBOL, quantity):

    if RUN_TYPE == 'TEST':
        print('***********Test mode, no trade executed')
        return

    try:
        buy_order = client.order_market_buy(symbol=SYMBOL, quantity=quantity)
        print('***********Order executed')
        logging.info('Order executed' + str(buy_order))
        print('Buy order executed. ID: {}, symbol: {}, Quantity: {}, Price: {}'.format(
               buy_order['orderId'], buy_order['symbol'], buy_order['executedQty'], buy_order['fills'][0]['price']
              ))
        return buy_order
    except Exception as e:
        print('***********Order failed')
        logging.info('Order failed' + str(e))
        print('Order failed: {}'.format(e))
        return None


def execute_sell_order(trade):

    try:

        # Execute sell order
        try:
            sell_order = client.order_market_sell(
                symbol=trade['symbol'],
                quantity=trade['quantity']
            )
        except: 
            print('Quanity not enough to sell, smaller quantity will be sold')
            sell_order = client.order_market_sell(
                 symbol=trade['symbol'],
                 quantity=str(float(trade['quantity']) * 0.9)
                )
        print('***********Sell Order executed')
        logging.info('Order executed' + str(sell_order))
        print('Sell order executed. ID: {}, symbol: {}, Quantity: {}, Price: {}'.format(
            sell_order['orderId'], sell_order['symbol'], sell_order['executedQty'], sell_order['fills'][0]['price']
              ))
        return sell_order 

    except Exception as e:
        print('***********Order failed')
        logging.info('Order failed' + str(e))
        print('Order failed: {}'.format(e))
        raise Exception('Order failed: {}'.format(e))



def check_balance(SYMBOL):
    balance = client.get_asset_balance(asset=SYMBOL)
    return float(balance['free'])


# ------------------------
# Handle data Functions 
# ------------------------

def init_data(SYMBOL):

    url = f'https://api.binance.com/api/v1/klines?symbol={SYMBOL}&interval=5m&limit={LOOKBACK}'

    r = requests.get(url)

    data_request = r.json()

    df = pd.DataFrame(data_request)

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


# ---------------------------------
# Tracking functions 
# ---------------------------------

def init_data_tracking(): 
    
        # Initiate data tracking
        data = {
        'orders': [],
        'open_trades': [],
        'closed_trades': [],
        'balance': [],
        'performance_metrics': {},
        }
    
        return data


def save_to_csv(data):
    # Save orders to CSV
    if data['orders']:
        with open('orders.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data['orders'][0].keys())
            writer.writeheader()
            writer.writerows(data['orders'])

    # Save open trades to CSV
    if data['open_trades']:
        with open('open_trades.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data['open_trades'][0].keys())
            writer.writeheader()
            writer.writerows(data['open_trades'])

    # Save closed trades to CSV
    if data['closed_trades']:
        with open('closed_trades.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data['closed_trades'][0].keys())
            writer.writeheader()
            writer.writerows(data['closed_trades'])

    # Save balance to CSV
    if data['balance']:
        with open('balance.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data['balance'][0].keys())
            writer.writeheader()
            writer.writerows(data['balance'])

    # Save performance metrics to CSV
    if data['performance_metrics']:
        with open('performance_metrics.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data['performance_metrics'].keys())
            writer.writeheader()
            writer.writerow(data['performance_metrics'])

    return print('Data saved to CSV')


def calculate_metrics():
    global data

    closed_trades = data['closed_trades']
    if not closed_trades:
        return

    # Initialize metrics
    num_winning_trades = 0
    num_losing_trades = 0
    total_profit = 0
    total_loss = 0
    returns = []

    # Calculate profits/losses and returns
    for trade in closed_trades:
        profit_loss = (float(trade['exit_price']) - float(trade['entry_price'])) * float(trade['quantity'])
        trade_value = float(trade['entry_price']) * float(trade['quantity'])

        if profit_loss > 0:
            num_winning_trades += 1
            total_profit += profit_loss
        else:
            num_losing_trades += 1
            total_loss += abs(profit_loss)

        returns.append(profit_loss / trade_value)

    # Calculate metrics
    profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else float('inf')
    if len(returns) > 1:
        max_drawdown = max([j-i for i, j in zip(returns[:-1], returns[1:])])
    else:
        max_drawdown = 0

    # Store metrics in data dictionary
    data['performance_metrics'] = {
        'winning_trades': num_winning_trades,
        'losing_trades': num_losing_trades,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
    }

    save_to_csv(data)

    return print('Metrics calculated and saved to CSV')


def update_counter(trade):
    trade['candle_counter'] += 2
    save_to_csv(data)


def track_buy_order(order):
    global data
    
    # Add the order to the list of orders
    data['orders'].append({
        'timestamp': datetime.datetime.now(),
        'type': 'buy',
        'symbol': order['symbol'],
        'quantity': order['executedQty'],
        'price': order['fills'][0]['price'],
        'status': 'filled',
        'order_id': order['orderId']
    })

    # Add the trade to the list of open trades
    data['open_trades'].append({
        'trade_id': order['orderId'],
        'symbol': order['symbol'],
        'entry_timestamp': datetime.datetime.now(),
        'entry_price': order['fills'][0]['price'],
        'quantity': order['executedQty'],
        'current_price': order['fills'][0]['price'],
        'current_profit_loss': 0,
        'candle_counter': 0
    })

    # Save the trade execution details and updated open trades to CSV
    save_to_csv(data)

    return print('Buy Order Tracked')


def track_sell_order(trade,sell_order):
    global data

    print('trade to be removed from open trades: ', trade)

    # Remove the trade from the dictionary data 
    data['open_trades'] = [i for i in data['open_trades'] if i['trade_id'] != trade['trade_id']]

    # Get sell order price 
    sell_order_price = sell_order['fills'][0]['price']

    # Add the trade to the list of closed trades
    data['closed_trades'].append({
        'trade_id': trade['trade_id'],
        'symbol': trade['symbol'],
        'entry_timestamp': trade['entry_timestamp'],
        'entry_price': trade['entry_price'],
        'exit_price': sell_order_price,
        'quantity': trade['quantity'],
        'exit_timestamp': datetime.datetime.now(),
    })

    # Save the trade execution details and updated open and closed trades to CSV
    save_to_csv(data)

    return print('Sell Order Tracked')


def get_price(symbol):

    url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        price = float(data['price'])
        return price
    else:
        raise Exception('Error retrieving price')


def update_balance():
    global data

    # Get current BTC balance
    btc_balance = check_balance('BTC')

    # Get current USDT balance
    usdt_balance = check_balance('USDT')

    # Convert BTC balance to USDT using the current price
    btc_usdt_price = get_price(SYMBOL)  # Replace 'BTCUSDT' with the appropriate symbol
    btc_usdt_balance = btc_balance * btc_usdt_price

    # Calculate the total balance
    total_balance = usdt_balance + btc_usdt_balance

    # Update balance in data dictionary
    data['balance'].append({
        'timestamp': datetime.datetime.now(),
        'balance': total_balance,
    })

    # Save balance to CSV
    with open('balance.csv', 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data['balance'][0].keys())
        writer.writerow(data['balance'][-1])

    return print('Balance Updated')


def start_update_balance_scheduler(update_frequency):
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_balance, 'interval', minutes=update_frequency)
    scheduler.start()

    return print('Balance Updated')


# ---------------------
# Webshoket functions 
# ---------------------

def on_message(ws, message,df):

    data_binance = json.loads(message)
    candlestick = data_binance['k']

    if candlestick['x']:

        print('***********Candle completed. Analysis begins')
        
        # Get initial time before analysis 
        init_time_check = time.time()

        if data['open_trades']:
            print('***********Get open trades')
            # Loop through all open trades
            for trade in data['open_trades']:
                update_counter(trade)
            
            for trade in data['open_trades']:
                if trade['candle_counter'] >= 2:
                    print('Trade achieved 5 candles. Execute sell order')
                    sell_order = None
                    sell_order  = execute_sell_order(trade)
                    
                    if sell_order:
                        # Track the sell order
                        print('Track sell order')
                        track_sell_order(trade, sell_order)
                        print('Sell order tracked')
                        
                        print('Update the balance')
                        update_balance()
                        print('Balance updated')
                        
                        print('Calculate metrics')
                        calculate_metrics()
                        print('Metrics calculated')

                        # Send message to Telegram
                        send_message_to_telegram('Sell order executed')
        else:
              print("No open trades yet")


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

        # Get signal
        if RUN_TYPE == 'TEST_TRADE':
            signal = signal_func()

        print('***********Signal: ', signal)

        # Get current usdt balance 
        print('***********Get current usdt balance')
        usdt_balance = check_balance('USDT')
        print('***********USDT Balance: ', usdt_balance)

        # Execute Trade 
        if signal:
            
            if usdt_balance > QUANTITY_PER_TRADE or len(data['open_trades']) > LIMIT_OPEN_TRADES:

                print('***********Execute Trade')

                # Get balance
                balance_before_trade = check_balance('USDT')
                print('***********Balance before trade: ', balance_before_trade)
                logging.info('Balance before trade:' + str(balance_before_trade))

                # Get quantity
                quantity = quantiy_trade(df, QUANTITY_PER_TRADE, step_size)

                # Execute trade
                buy_order = execute_buy_order(SYMBOL,quantity=quantity)

                # Get final time after analysis
                final_time_check = time.time()
                
                # Get analysis duration in seconss 
                execution_time = final_time_check - init_time_check

                print('***********Buy order Execution Time: ', execution_time)

                # Update balance
                balance_after_trade = check_balance('USDT')
                print('***********Balance after trade: ', balance_after_trade)
                logging.info('Balance after trade:' + str(balance_after_trade))

                # Track buy order
                track_buy_order(buy_order)

                # Update balance
                update_balance()

                # Calculate metrics
                calculate_metrics()

                # Send message to Telegram
                send_message_to_telegram('Buy order executed')

            else:
                if usdt_balance < QUANTITY_PER_TRADE:
                    print('***********Not enough balance to execute trade')
                    logging.info('Not enough balance to execute trade')
                if len(data['open_trades']) > LIMIT_OPEN_TRADES:
                    print('***********Limit open trades reached')
                    logging.info('Limit open trades reached')

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
             f'{SYMBOL_webshocket}@kline_{TIMEFRAME}'
        ],
        'id': 1
    }
    ws.send(json.dumps(subscribe_data))


if __name__ == '__main__':

    # Init data
    print('***********Initializing data')
    logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    
    # Init data
    df = init_data(SYMBOL)

    # Get step size
    step_size = get_step_size(SYMBOL)

    # Run type
    print('***********Run type: ', RUN_TYPE)

    # Init balance
    print('***********Initializing balance')
    logging.info('Initializing balance')
    
    balance = check_balance('USDT')

    print('***********Balance: ', balance)

    # Initialize data dictionary
    print('***********Initializing data dictionary')
    data = init_data_tracking() 

    # Init client
    print('***********Initializing binance client')
    logging.info('Initializing binance client')

    client = Client(api_key, api_secret)

    # Init telegram client
    print('***********Initializing telegram client')
    logging.info('Initializing telegram client')

    client_telegram = connect_tg()
    client_telegram.connect()

    # Send init message to telegram
    send_message_to_telegram('Bot started')

    # Start balance update scheduler
    print('***********Starting balance update scheduler')
    logging.info('Starting balance update scheduler')
    start_update_balance_scheduler(UPDATE_FREQUENCY)

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


