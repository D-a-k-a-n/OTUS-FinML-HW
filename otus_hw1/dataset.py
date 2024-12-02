from pathlib import Path
import os
import pandas as pd
import yfinance as yf
import ccxt
import typer  # type: ignore
from loguru import logger  # type: ignore
from tqdm import tqdm  # type: ignore

from config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# Очистка данных S&P 500
def clean_sp500_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает DataFrame с некорректно отформатированными строками и столбцами.

    Параметры:
        df (pd.DataFrame): Исходный DataFrame для очистки.

    Возвращает:
        pd.DataFrame: Очищенный DataFrame с правильными заголовками столбцов и типами данных.
    """
    # Шаг 1: Упрощение MultiIndex столбцов
    df.columns = df.columns.get_level_values(0)  # Retain only the 'Price' level

    # Шаг 2: Сброс индекса для преобразования 'Date' из индекса в столбец
    df = df.reset_index().rename(columns={'index': 'Date'})
    
    # Шаг 3: Преобразовать типы данных в правильные форматы
    try:
        df["Date"] = pd.to_datetime(df["Date"])  # Конвертируем столбец Date в формат datetime
        df[["Adj Close", "Close", "High", "Low", "Open"]] = df[
            ["Adj Close", "Close", "High", "Low", "Open"]
        ].astype(float)  # Конвертируем числовые столбцы в тип float
        df["Volume"] = df["Volume"].astype(int)  # Конвертируем столбец Volume в тип int
    except KeyError as e:
        raise ValueError(f"Отсутствуют необходимые столбцы: {e}")
    except ValueError as e:
        raise ValueError(f"Ошибка преобразования типов данных: {e}")

    return df


# Функция для сохранения данных в Parquet формате
def save_as_parquet(df: pd.DataFrame, output_path: Path):
    """
    Сохраняет DataFrame в формате Parquet.

    Параметры:
        df (pd.DataFrame): Данные для сохранения.
        output_path (Path): Путь для сохранения файла Parquet.
    """
    try:
        df.to_parquet(output_path, index=False)
    except Exception as e:
        logger.warning(f"Ошибка сохранения в Parquet: {e}")


# Функции загрузки данных
def download_sp500_data(output_dir: Path, processed_dir: Path):
    """
    Загрузка данных по компаниям из S&P 500, сохранение в output_dir (CSV) 
    и processed_dir (Parquet).
    """
    logger.info("Загрузка данных S&P 500...")
    sp500_tickers_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    sp500_tickers = pd.read_csv(sp500_tickers_url)['Symbol'].tolist()

    for ticker in tqdm(sp500_tickers, desc="Загрузка S&P 500"):
        try:
            stock_data = yf.download(ticker, period="1y", interval="1d", progress=False)
            if not stock_data.empty:
                stock_data = clean_sp500_data(stock_data)
                # Сохранение в CSV
                stock_data.to_csv(output_dir / f"{ticker}.csv", index=False)
                # Сохранение в Parquet
                save_as_parquet(stock_data, processed_dir / f"{ticker}.parquet")
        except Exception as e:
            logger.warning(f"Ошибка загрузки данных для {ticker}: {e}")
    logger.success("Данные S&P 500 успешно загружены.")


def download_crypto_data(output_dir: Path, processed_dir: Path):
    """
    Загрузка данных по криптовалютам, сохранение в output_dir (CSV) 
    и processed_dir (Parquet).
    """
    logger.info("Загрузка данных по криптовалютам...")
    cryptos = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
    exchange = ccxt.binance()

    for crypto in tqdm(cryptos, desc="Загрузка криптовалют"):
        try:
            ohlcv = exchange.fetch_ohlcv(crypto, timeframe='1d', limit=365)
            df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['Date'], unit='ms')
            # Сохранение в CSV
            df.to_csv(output_dir / f"{crypto.replace('/', '_')}.csv", index=False)
            # Сохранение в Parquet
            save_as_parquet(df, processed_dir / f"{crypto.replace('/', '_')}.parquet")
        except Exception as e:
            logger.warning(f"Ошибка загрузки данных для {crypto}: {e}")
    logger.success("Данные по криптовалютам успешно загружены.")


# Основная команда приложения
@app.command()
def main(
    # ---- Параметры с дефолтными путями ----
    sp500_output_dir: Path = RAW_DATA_DIR / "sp500_data",
    crypto_output_dir: Path = RAW_DATA_DIR / "crypto_data",
    sp500_interim_dir: Path = INTERIM_DATA_DIR / "sp500_data",
    crypto_interim_dir: Path = INTERIM_DATA_DIR / "crypto_data",
):
    """
    Главная функция для загрузки данных по S&P 500 и криптовалютам.
    """
    # Создание выходных директорий, если их нет
    sp500_output_dir.mkdir(parents=True, exist_ok=True)
    crypto_output_dir.mkdir(parents=True, exist_ok=True)
    sp500_interim_dir.mkdir(parents=True, exist_ok=True)
    crypto_interim_dir.mkdir(parents=True, exist_ok=True)

    # Загрузка данных
    download_sp500_data(sp500_output_dir, sp500_interim_dir)
    download_crypto_data(crypto_output_dir, crypto_interim_dir)

    logger.success("Все данные успешно загружены и сохранены.")

if __name__ == "__main__":
    app()