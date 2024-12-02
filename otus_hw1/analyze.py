import os
from pathlib import Path
import pandas as pd
from loguru import logger  # Для логирования
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR
import typer
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go

app = typer.Typer()

# Функция для создания графиков японских свечей с выделением аномалий
def save_ohlc_plot_with_anomalies(data: pd.DataFrame, ticker: str, outliers: pd.DataFrame, output_dir: Path):
    """
    Создает и сохраняет график японских свечей с выделением аномалий.

    Параметры:
        data (pd.DataFrame): Данные OHLC.
        ticker (str): Название тикера.
        outliers (pd.DataFrame): Данные выбросов.
        output_dir (Path): Директория для сохранения графиков.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, outlier in outliers.iterrows():
        idx = data[data['Date'] == outlier['Date']].index[0]
        start_idx = max(idx - 25, 0)  # Обрезаем 50 свечей: 25 до и 25 после выброса
        end_idx = min(idx + 25, len(data) - 1)

        subset = data.iloc[start_idx:end_idx]

        # Построение графика
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=subset['Date'],
                    open=subset['Open'],
                    high=subset['High'],
                    low=subset['Low'],
                    close=subset['Close']
                )
            ]
        )

        # Добавляем выделение выброса красным квадратом
        fig.add_shape(
            type="rect",
            x0=outlier["Date"],
            x1=outlier["Date"],
            y0=subset["Low"].min(),
            y1=subset["High"].max(),
            line=dict(color="red", width=2),
        )

        fig.update_layout(
            title=f"{ticker}: Anomaly at {outlier['Date']}",
            xaxis_title="Date",
            yaxis_title="Price",
        )

        # Сохраняем график
        output_path = output_dir / f"{ticker}_{outlier['Date']}.png"
        fig.write_image(str(output_path))
        logger.info(f"График сохранен: {output_path}")

# Функция для сохранения HTML-отчета
def save_html_report(report: list, output_path: Path):
    """
    Сохраняет отчет в формате HTML.

    Параметры:
        report (list): Данные отчета.
        output_path (Path): Путь для сохранения HTML-файла.
    """
    try:
        env = Environment(loader=FileSystemLoader("reports/templates"))
        template = env.get_template("report_template.html")
        html_content = template.render(report=report)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"HTML-отчет сохранен в {output_path}.")
    except Exception as e:
        logger.error(f"Ошибка сохранения HTML-отчета: {e}")

# Функция анализа данных
def check_data_quality(file_path: Path, save_dir: Path, figures_dir: Path) -> dict:
    """
    Анализирует данные на пропуски и выбросы, создает графики с японскими свечами.

    Параметры:
        file_path (Path): Путь к файлу данных.
        save_dir (Path): Директория для сохранения обработанных данных.
        figures_dir (Path): Директория для сохранения графиков.

    Возвращает:
        dict: Результаты анализа.
    """
    analysis_result = {"file": str(file_path.name)}  # Сохраняем имя файла
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        logger.warning(f"Ошибка загрузки файла {file_path}: {e}")
        analysis_result["error"] = str(e)
        return analysis_result

    column_to_check = "Close" if "Close" in data.columns else None
    if not column_to_check:
        logger.warning(f"Столбец 'Close' отсутствует в файле {file_path}. Пропущено.")
        analysis_result["status"] = "skipped"
        return analysis_result

    # Проверка на пропуски
    missing_values = data[column_to_check].isnull().sum()
    analysis_result["missing_values"] = missing_values
    if missing_values > 0:
        data[column_to_check].fillna(data[column_to_check].mean(), inplace=True)
        analysis_result["missing_filled"] = True

    # Поиск выбросов
    Q1 = data[column_to_check].quantile(0.25)
    Q3 = data[column_to_check].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[
        (data[column_to_check] < lower_bound) | (data[column_to_check] > upper_bound)
    ]
    analysis_result["outliers_count"] = len(outliers)

    if not outliers.empty:
        ticker = file_path.stem
        save_ohlc_plot_with_anomalies(data, ticker, outliers, figures_dir / ticker)

    # Сохранение очищенных данных
    save_path = save_dir / file_path.name
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(save_path, index=False)
        analysis_result["status"] = "cleaned"
    except Exception as e:
        logger.error(f"Ошибка сохранения файла {save_path}: {e}")
        analysis_result["error"] = str(e)

    return analysis_result

# Основная функция анализа качества данных
def analyze_data_quality(input_dir: Path, output_dir: Path, report_file: Path, figures_dir: Path, top_5: bool = False):
    """
    Выполняет анализ качества данных.

    Параметры:
        input_dir (Path): Путь к директории с файлами данных.
        output_dir (Path): Директория для сохранения обработанных файлов.
        report_file (Path): Путь для сохранения отчета.
        figures_dir (Path): Директория для сохранения графиков.
        top_5 (bool): Обработать только топ-5 тикеров.
    """
    logger.info(f"Анализ качества данных в директории {input_dir}...")
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"В директории {input_dir} нет файлов CSV для анализа.")
        return

    if top_5:
        ticks = {"AAPL.csv", "AMZN.csv", "NFLX.csv", "NVDA.csv", "GOOGL.csv"}
        csv_files = [file for file in csv_files if file.name in ticks]

    report = []
    for file_path in csv_files:
        try:
            result = check_data_quality(file_path, output_dir, figures_dir)
            report.append(result)
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {file_path}: {e}")

    save_html_report(report, report_file)

# Основная команда CLI
@app.command()
def main(
    sp500_input_dir: Path = RAW_DATA_DIR / "sp500_data",
    crypto_input_dir: Path = RAW_DATA_DIR / "crypto_data",
    sp500_output_dir: Path = PROCESSED_DATA_DIR / "sp500_data",
    crypto_output_dir: Path = PROCESSED_DATA_DIR / "crypto_data",
    sp500_report_file: Path = REPORTS_DIR / "sp500_report.html",
    crypto_report_file: Path = REPORTS_DIR / "crypto_report.html",
    sp500_figures_dir: Path = FIGURES_DIR / "sp500",
    crypto_figures_dir: Path = FIGURES_DIR / "crypto",
    top_5: bool = typer.Option(
        True,
        help="Обработать только топ-5 тикеров S&P 500."
    )
):
    """
    Главная функция для анализа качества данных.

    Параметры:
        sp500_input_dir (Path): Директория данных S&P 500.
        crypto_input_dir (Path): Директория данных криптовалют.
        sp500_output_dir (Path): Директория для сохранения обработанных данных S&P 500.
        crypto_output_dir (Path): Директория для сохранения обработанных данных криптовалют.
        sp500_report_file (Path): Путь для сохранения отчета по S&P 500.
        crypto_report_file (Path): Путь для сохранения отчета по криптовалютам.
        sp500_figures_dir (Path): Директория для сохранения графиков S&P 500.
        crypto_figures_dir (Path): Директория для сохранения графиков криптовалют.
        top_5 (bool): Обработать только топ-5 тикеров.
    """
    analyze_data_quality(sp500_input_dir, sp500_output_dir, sp500_report_file, sp500_figures_dir, top_5)
    analyze_data_quality(crypto_input_dir, crypto_output_dir, crypto_report_file, crypto_figures_dir)

    logger.success("Анализ качества данных завершен.")

if __name__ == "__main__":
    app()
