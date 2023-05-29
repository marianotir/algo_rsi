

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
import configparser
import dash_bootstrap_components as dbc


# -------------------
# Define functions 
# -------------------

def read_csv_with_headers(filename, headers):
    try:
        df = pd.read_csv(filename)
        if df.empty:
            df = pd.DataFrame(columns=headers)
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=headers)
    
open_trades_headers = ['trade_id', 'symbol', 'entry_timestamp', 'entry_price', 'quantity', 
                      'candle_counter']


# Update data function
def update_data():
    global open_trades, metrics, balance, backtest,backtest_metrics,balance_fig,backtest_fig,asset_exposure_fig
    open_trades = read_csv_with_headers('open_trades.csv',open_trades_headers)
    metrics = pd.read_csv('performance_metrics.csv')
    balance = pd.read_csv('balance.csv')
    backtest = pd.read_csv('backtest_results.csv')
    backtest_metrics = pd.read_csv('backtest_metrics.csv')
    balance['timestamp'] = pd.to_datetime(balance['timestamp'])
    backtest['timestamp'] = pd.to_datetime(backtest['timestamp'])
    if len(open_trades) > 0:
        open_trades['entry_timestamp'] = pd.to_datetime(open_trades['entry_timestamp'])

    
    # balance fig 
    balance_fig = px.line(balance, x='timestamp', y='balance', title='Total Current Balance')
    balance_fig.update_layout(
    template='plotly_dark',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    )
    
    # backtest fig
    backtest_fig = px.line(backtest, x='timestamp', y='cum_strategy_return', title='Backtest')
    backtest_fig.update_xaxes(title_text='Date')
    backtest_fig.update_yaxes(title_text='Return')
    backtest_fig.update_layout(
    template='plotly_dark',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    )

    # define asset exposure figure
    asset_exposure_fig = px.pie(open_trades, names='symbol', title='Assets Exposure')
    asset_exposure_fig.update_layout(
    template='plotly_dark',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)',
    )


# ------------------------
# Get initial variables 
# ------------------------

# Create a config parser
config = configparser.ConfigParser()

# Read the config file
config.read('config.ini')

# Get the update interval
update_interval = config.getint('settings', 'update_interval')


# ---------------------
# Initialize variables
# ---------------------

# Initialize data
update_data()


# ---------------------
# Define Dash app
# ---------------------

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

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
                figure=asset_exposure_fig
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
                id='backtest_metrics_table',
                columns=[{"name": i, "id": i} for i in backtest_metrics.columns],
                data=backtest_metrics.to_dict('records'),
                style_table={'overflowX': 'auto', 'maxHeight': '300px'},
                style_cell={'height': 'auto'},
            )
        ]),
    ]),
    dcc.Interval(id='interval-component', interval=update_interval*60*1000)  # in milliseconds
])

# Callbacks:
@app.callback(
    Output('balance_graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_balance_fig(n):
    update_data()
    return balance_fig

@app.callback(
    Output('open_trades_table', 'data'),
    Input('interval-component', 'n_intervals')
)
def update_open_trades(n):
    update_data()
    return open_trades.to_dict('records')

@app.callback(
    Output('asset_exposure', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_asset_exposure(n):
    update_data()
    return asset_exposure_fig

@app.callback(
    Output('metrics_table', 'data'),
    Input('interval-component', 'n_intervals')
)
def update_metrics(n):
    update_data()
    return metrics.to_dict('records')


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=False)
