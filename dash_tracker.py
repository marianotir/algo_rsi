

# ---------------------
# Import libraries
# ---------------------

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import dash_table
import pandas as pd
import plotly.express as px
import os


# -------------------
# Define functions 
# -------------------

def read_csv_with_headers(filename, headers):
    if os.stat(filename).st_size > 0:  # Check if file is not empty
        return pd.read_csv(filename)
    else:  # If file is empty, return empty dataframe with headers
        return pd.DataFrame(columns=headers)
    
open_trades_headers = ['trade_id', 'symbol', 'entry_timestamp', 'entry_price', 'quantity', 
                      'candle_counter']


# ---------------------
# Initiate variables 
# ---------------------

# Load data
open_trades = read_csv_with_headers('open_trades.csv', open_trades_headers)
metrics = pd.read_csv('performance_metrics.csv')
balance = pd.read_csv('balance.csv')
backtest = pd.read_csv('backtest_results.csv')
backtest_metrics = pd.read_csv('backtest_metrics.csv')


balance['timestamp'] = pd.to_datetime(balance['timestamp'])
open_trades['entry_timestamp'] = pd.to_datetime(open_trades['entry_timestamp'])
backtest['timestamp'] = pd.to_datetime(backtest['timestamp'])

# define figures
balance_fig = px.line(balance, x='timestamp', y='balance', title='Total Current Balance')
backtest_fig = px.line(backtest, x='timestamp', y='cum_strategy_return', title='Backtest')
# Define x and y tltles as x date and y return 
backtest_fig.update_xaxes(title_text='Date')
backtest_fig.update_yaxes(title_text='Return')


# ---------------------
# Define Dash app
# ---------------------

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    dcc.Tabs(id='Estrategy', value='tab-1', children=[
        dcc.Tab(label='Tracker', value='tab-1', children=[
            html.H1('Algo Trading Price Reversal Strategy Tracker'),
            dcc.Graph(
                id='balance_graph',
                figure=balance_fig
            ),
            html.H2('Open Trades'),
            dash_table.DataTable(
                id='open_trades_table',
                columns=[{"name": i, "id": i} for i in open_trades.columns],
                data=open_trades.to_dict('records'),
                style_table={'overflowX': 'auto', 'maxHeight': '300px'},
                style_cell={'height': 'auto'},
            ),
            dcc.Graph(
                id='asset_exposure',
                figure=px.pie(open_trades, names='symbol', title='Assets Exposure')
            ),
            html.H2('Metrics'),
            dash_table.DataTable(
                id='metrics_table',
                columns=[{"name": i, "id": i} for i in metrics.columns],
                data=metrics.to_dict('records'),
                style_table={'overflowX': 'auto', 'maxHeight': '300px'},
                style_cell={'height': 'auto'},
            )
        ]),
        dcc.Tab(label='Backtest', value='tab-2', children=[
            dcc.Graph(
                id='backtest_graph',
                figure=backtest_fig
            ),
            html.H2('Backtest Metrics'),
            dash_table.DataTable(
                id='metrics_table',
                columns=[{"name": i, "id": i} for i in backtest_metrics.columns],
                data=backtest_metrics.to_dict('records'),
                style_table={'overflowX': 'auto', 'maxHeight': '300px'},
                style_cell={'height': 'auto'},
            )
        ]),
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)