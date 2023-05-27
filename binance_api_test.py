

import configparser
import requests

# Read config file
config = configparser.ConfigParser()
config.read('config.ini')


api_key = config['binance']['key']
api_secret = config['binance']['secret']

from binance.spot import Spot

client =  Spot()

# Get server timestamp
print(client.time())
# Get klines of BTCUSDT at 1m interval
print(client.klines("BTCUSDT", "1m"))
# Get last 10 klines of BNBUSDT at 1h interval
print(client.klines("BNBUSDT", "1h", limit=10))

# API key/secret are required for user data endpoints
client = Spot(api_key=api_key, api_secret=api_secret)

# Get account and balance information
print(client.account())

# check quantity for tokens which are not value cero
for balance in client.account()['balances']:
    if float(balance['free']) > 0:
        print(balance)


# Post a new order
params = {
    'symbol': 'ETHUSDT',
    'side': 'SELL',
    'type': 'LIMIT',
    'timeInForce': 'GTC',
    'quantity': 0.00507706,
    'price': 9500
}

response = client.new_order(**params)
print(response)



import os

from binance.client import Client

import configparser
import requests

# Read config file
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['binance']['key']
api_secret = config['binance']['secret']

client = Client(api_key, api_secret)


# get balances for all assets & some account information
print(client.get_account())

# get balance for a specific asset only (BTC)
print(client.get_asset_balance(asset='ETH'))

# get latest price from Binance API
btc_price = client.get_symbol_ticker(symbol="BTCUSDT")
# print full output (dictionary)
print(btc_price)
print(btc_price["price"])

from time import sleep

from binance.streams import ThreadedWebsocketManager

def btc_trade_history(msg):
    ''' define how to process incoming WebSocket messages '''
    if msg['e'] != 'error':
        print(msg['c'])
        btc_price['last'] = msg['c']
        btc_price['bid'] = msg['b']
        btc_price['last'] = msg['a']
        btc_price['error'] = False
    else:
        btc_price['error'] = True
    
bsm = ThreadedWebsocketManager()
bsm.start()

bsm.start_symbol_ticker_socket(callback=btc_trade_history, symbol='BTCUSDT')


# get timestamp of earliest date data is available
timestamp = client._get_earliest_valid_timestamp('BTCUSDT', '1m')
print(timestamp)

# request historical candle (or klines) data
bars = client.get_historical_klines('BTCUSDT', '1m', timestamp, limit=1000)

import pandas as pd 

for line in bars:
    del line[5:]


btc_df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close'])
btc_df.set_index('date', inplace=True)
print(btc_df.head())

# transformt date to datetime object
btc_df.index = pd.to_datetime(btc_df.index, unit='ms')
btc_df


# use get request with this website https://api.binance.com/api/v1/klines?symbol=YFIUSDT&interval=1m&limit=1000

import requests

symbol = 'BTCUSDT'
url = f'https://api.binance.com/api/v1/klines?symbol={symbol}&interval=1m&limit=1000'

r = requests.get(url)

data = r.json()

import pandas as pd

df = pd.DataFrame(data)

df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']

df.head()

# transformt open_time to datetime object
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df


# Get the step size
step_size = float('0.00010000')

# Define the quantity you want to sell
quantity = 0.00507706

# Get the minimum notional value and the step size
min_notional = float('10.00000000')
step_size = float('0.00010000')

# Get the current price of ETH
eth_price = float(client.get_symbol_ticker(symbol='ETHUSDT')['price'])

# Calculate the minimum quantity you can sell at the current price to meet the min_notional
min_quantity = min_notional / eth_price

# If the quantity is less than min_quantity, set it to min_quantity
if quantity < min_quantity:
    quantity = min_quantity

# Adjust the quantity to meet the step size requirement
quantity = quantity - (quantity % step_size)

# Now try to make the order
market_order = client.order_market_sell(symbol='ETHUSDT', quantity=0.01)

# If the order was successful, this will print the order dict
print(market_order)

# If the order was unsuccessful, this will raise an exception
print(market_order['msg'])

# market order by 
market_order = client.order_market_buy(symbol='ETHUSDT', quantity=10)

market_order

# check eth quantity in the user 
client.get_asset_balance(asset='USDT')


symbol = 'ETHUSDT'
quantity = 0.01  # replace this with the amount of ETH you want to buy

# Place a market buy order
try:
    market_order = client.order_market_buy(symbol=symbol, quantity=quantity)
    print(market_order)
except Exception as e:
    print(e)

client.get_asset_balance(asset='ETH')



btc_price = float(client.get_symbol_ticker(symbol='BTCUSDT')['price'])

step_size = 0.0001

# I want to buy 100 USD worth of BTC
usdt_quantity = 20

# Calculate the quantity of BTC to buy
quantity = float(usdt_quantity / btc_price)

# Adjust the quantity to meet the step size requirement
quantity = quantity - (quantity % step_size)

# Place the order
market_order = client.order_market_buy(symbol='BTCUSDT', quantity=quantity)


client.get_asset_balance(asset='BTC')

client.order_market_sell(symbol='BTCUSDT', quantity=0.0014)





from binance.client import Client

# Instantiate a Client object with your API key and secret
client = Client(api_key, api_secret)

# Define the symbol, quantity, and price you want to sell at
symbol = 'ETHUSDT'
quantity = 0.00507706
price = '4000.00'  # replace this with your desired sell price

# Place a limit sell order
order = client.order_limit_sell(
    symbol=symbol,
    quantity=quantity,
    price=price
)



import math

step_size = float('0.00010000')  # replace with the actual step size
quantity = 0.00507706  # your original quantity

# Adjust quantity to match step size
quantity = math.floor(quantity / step_size) * step_size

# Convert back to string for the Binance API
quantity = '{:.8f}'.format(quantity)

# Now place the order
order = client.order_limit_sell(
    symbol=symbol,
    quantity=0.005,
    price=price
)


orders = client.get_open_orders()

for order in orders:
    print(order)


# cancel order
symbol = 'ETHUSDT'
order_id = '13473759483'  # replace this with the actual order ID

# Cancel the order
result = client.cancel_order(
    symbol=symbol,
    orderId=order_id
)

print(result)