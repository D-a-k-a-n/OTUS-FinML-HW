from dash import Dash, dcc, _dash_renderer
import dash_mantine_components as dmc
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

_dash_renderer._set_react_version("18.2.0")


def load_data(data_dir: Path):
    """
    Load S&P 500 and cryptocurrency data from specified directory.
    """
    sp500_dir = data_dir / "sp500_data"
    crypto_dir = data_dir / "crypto_data"

    sp500_data = {file.stem: pd.read_parquet(file) for file in sp500_dir.glob("*.parquet")}
    crypto_data = {file.stem: pd.read_parquet(file) for file in crypto_dir.glob("*.parquet")}

    return sp500_data, crypto_data


def create_ohlc_chart(data: pd.DataFrame, title: str):
    """
    Create an OHLC chart from the given data.
    """
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data["timestamp"] if "timestamp" in data.columns else data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name=title,
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis=dict(
            rangeslider=dict(visible=True),  # Включаем ползунок для удобства
            range=[data["Date"].iloc[-90], data["Date"].iloc[-1]],  # Начальный диапазон
        ),
    )

    return fig


def generate_tabs(data: dict, tab_value: str):
    """
    Generate a set of tabs for a category (e.g., S&P 500 or Crypto).
    """
    return dmc.Tabs(
        value="0",
        children=[
            dmc.TabsList([dmc.TabsTab(ticker, value=str(idx)) for idx, ticker in enumerate(data)])
        ]
        + [
            dmc.TabsPanel(
                dcc.Graph(figure=create_ohlc_chart(data[ticker], f"{ticker} OHLC")),
                value=str(idx),
            )
            for idx, ticker in enumerate(data)
        ],
    )


def run_dash_app(data_dir: Path, top_5: bool):
    """
    Run the Dash app for data visualization.
    """
    sp500_data, crypto_data = load_data(data_dir)
    # Extracting the 5 ticks from SP500
    if top_5:
        ticks = ["AAPL", "AMZN", "NFLX", "NVDA", "GOOGL"]
        sp500_data = {key: sp500_data[key] for key in ticks}
    # sp500_data = {key: value for i, (key, value) in enumerate(sp500_data.items()) if i < 5}

    app = Dash(__name__, suppress_callback_exceptions=True)

    # Wrap the layout with MantineProvider
    app.layout = dmc.MantineProvider(
        children=[
            dmc.Tabs(
                value="sp500",
                children=[
                    dmc.TabsList(
                        [
                            dmc.TabsTab("S&P 500", value="sp500"),
                            dmc.TabsTab("Crypto", value="crypto"),
                        ]
                    ),
                    dmc.TabsPanel(generate_tabs(sp500_data, "sp500"), value="sp500"),
                    dmc.TabsPanel(generate_tabs(crypto_data, "crypto"), value="crypto"),
                ],
            )
        ]
    )

    app.run(debug=True)
