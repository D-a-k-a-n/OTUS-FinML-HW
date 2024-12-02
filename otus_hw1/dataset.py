from pathlib import Path
import os
import pandas as pd
import yfinance as yf
import ccxt
import typer  # type: ignore
from loguru import logger  # type: ignore
from tqdm import tqdm  # type: ignore

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# Функции загрузки данных
def download_sp500_data(output_dir: Path):
    """
    Загрузка данных по компаниям из S&P 500 и сохранение в output_dir.
    """
    logger.info("Загрузка данных S&P 500...")
    sp500_tickers_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    sp500_tickers = pd.read_csv(sp500_tickers_url)['Symbol'].tolist()

    for ticker in tqdm(sp500_tickers, desc="Загрузка S&P 500"):
        try:
            stock_data = yf.download(ticker, period="1y", interval="1d", progress=False)
            if not stock_data.empty:
                stock_data.to_csv(output_dir / f"{ticker}.csv")
        except Exception as e:
            logger.warning(f"Ошибка загрузки данных для {ticker}: {e}")
    logger.success("Данные S&P 500 успешно загружены.")

def download_crypto_data(output_dir: Path):
    """
    Загрузка данных по криптовалютам и сохранение в output_dir.
    """
    logger.info("Загрузка данных по криптовалютам...")
    cryptos = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
    exchange = ccxt.binance()

    for crypto in tqdm(cryptos, desc="Загрузка криптовалют"):
        try:
            ohlcv = exchange.fetch_ohlcv(crypto, timeframe='1d', limit=365)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.to_csv(output_dir / f"{crypto.replace('/', '_')}.csv", index=False)
        except Exception as e:
            logger.warning(f"Ошибка загрузки данных для {crypto}: {e}")
    logger.success("Данные по криптовалютам успешно загружены.")

# Основная команда приложения
@app.command()
def main(
    # ---- Параметры с дефолтными путями ----
    sp500_output_dir: Path = RAW_DATA_DIR / "sp500_data",
    crypto_output_dir: Path = RAW_DATA_DIR / "crypto_data",
):
    """
    Главная функция для загрузки данных по S&P 500 и криптовалютам.
    """
    # Создание выходных директорий, если их нет
    sp500_output_dir.mkdir(parents=True, exist_ok=True)
    crypto_output_dir.mkdir(parents=True, exist_ok=True)

    # Загрузка данных
    download_sp500_data(sp500_output_dir)
    download_crypto_data(crypto_output_dir)

    logger.success("Все данные успешно загружены.")


if __name__ == "__main__":
    app()
