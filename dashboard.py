from classifier.lsd import *
import argparse

import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, callback, Output, Input
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from common import get_data, get_session_data, get_session_times_from_date_time


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
        className="app-div",
        children=[
            dbc.Container(
                className="header g-0",
                fluid=True,
                children=[
                    dbc.Row([
                        dbc.Col(html.H1("Lorentzian Indicator"), width=4),
                        dbc.Col(width=8, children=[
                            dbc.Row([
                                dbc.Col(children=[dcc.Dropdown(['2025-10-24'], '2025-10-24', id='dropdown-selection')], width=4),
                                dbc.Col([
                                    html.Label("Start Time: "),
                                    dbc.Input(id="start-time", value="11:30"),
                                ], width=4),
                                dbc.Col([
                                    html.Label("End Time: "),
                                    dbc.Input(id="end-time", value="18:30"),
                                ], width=4),
                            ])
                        ])
                    ]),
                    html.Hr(),
                ]),
            dbc.Container(
                className="body g-0",
                fluid=True,
                children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='graph-content'), width=12),
                    ])
                ]
            ),
        ],
    )


def run_analysis(df) -> pd.DataFrame:
    lc = LorentzianSpaceDistanceIndictor(df)
    return lc.lsd()


@callback(
    Output('graph-content', 'figure'),
    Input('dropdown-selection', 'value'),
    Input('start-time', 'value'),
    Input('end-time', 'value'),
)
def update_graph(date, start_time, end_time):
    csv_path = 'data/' + date + '.csv'
    df = run_analysis(get_data(csv_path))

    start_dt, end_dt = get_session_times_from_date_time('2025-10-24', start_time, end_time)
    df = get_session_data(df, start_dt, end_dt)

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_candlestick(x=df['ts_date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], row=1, col=1)
    fig.add_trace(go.Scatter(x=df['ts_date'], y=df['low'] - 0.25, text=df['predictions'], mode='text',
                             textposition='bottom center',
                             showlegend=False,
                             textfont=(dict(size=10, color='green'))), row=1, col=1)
    if 'yhat1' in df.columns:
        fig.add_trace(go.Scatter(x=df['ts_date'], y=df['yhat1'], mode="lines",
                                 marker=dict(color="yellow"), name="yhat1"), row=1, col=1)

    if 'start_long_trade' in df.columns and 'start_short_trade' in df.columns:
        fig.add_trace(go.Scatter(x=df['ts_date'], y=df['start_long_trade'], mode="markers",
                                 marker=dict(color="green", symbol="triangle-up", size=20), name="Long"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['ts_date'], y=df['start_short_trade'], mode="markers",
                                 marker=dict(color="red", symbol="triangle-down", size=20), name="Short"), row=1, col=1)

    fig.update_layout(autosize=True, height=1200, xaxis_rangeslider_visible=False)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="output file name", default="output.csv")
    parser.add_argument("-d", "--data", help="data file name", default="data/train.csv")
    args = parser.parse_args()

    app.run(debug=True, port=8080, host='0.0.0.0')
