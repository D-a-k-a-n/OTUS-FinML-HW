import os
from pathlib import Path
import pandas as pd
from loguru import logger  # Для логирования
from config import RAW_DATA_DIR, REPORTS_DIR, FIGURES_DIR
import typer
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

app = typer.Typer()


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


# Сохранение аномальных отрезков
def save_anomalies_visual_candlestick(df: pd.DataFrame, output_dir: Path):
    """
    Сохраняет графики аномальных отрезков в виде японских свечей с помощью Plotly.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    anomaly_types = {
        "Outlier_IQR": "IQR",
        "Outlier_LOF": "LOF",
        "Outlier_IsolationForest": "IsolationForest",
    }

    for anomaly_col, method in anomaly_types.items():
        anomalies = df[df[anomaly_col]]
        logger.info(f"Сохранение аномалий для метода {method}. Найдено {len(anomalies)} выбросов.")

        for i, idx in enumerate(
            anomalies.index[:3]
        ):  # Сохраним только 3 первые аномалии для каждого метода
            start = max(0, idx - 25)
            end = min(len(df), idx + 25)
            segment = df.iloc[start:end].copy()

            # Проверяем наличие столбца Date
            if "Date" not in df.columns:
                logger.error("Столбец 'Date' отсутствует в данных. Пропускаем.")
                continue

            # Преобразуем Date в datetime
            segment["Date"] = pd.to_datetime(segment["Date"])

            # Проверяем, находятся ли все необходимые столбцы
            required_columns = ["Open", "High", "Low", "Close"]
            if not all(col in segment.columns for col in required_columns):
                logger.warning(
                    f"Сегмент не содержит всех необходимых столбцов: {required_columns}. Пропускаем."
                )
                continue

            # Преобразуем индекс в дату
            try:
                idx_date = pd.to_datetime(df.loc[idx, "Date"])
            except KeyError:
                logger.warning(f"Индекс {idx} не найден в данных. Пропускаем.")
                continue

            # Проверяем, находится ли дата аномалии в сегменте
            if idx_date not in segment["Date"].values:
                logger.warning(f"Дата аномалии {idx_date} отсутствует в сегменте. Пропускаем.")
                continue

            # Построение японских свечей
            fig = go.Figure()

            fig.add_trace(
                go.Candlestick(
                    x=segment["Date"],
                    open=segment["Open"],
                    high=segment["High"],
                    low=segment["Low"],
                    close=segment["Close"],
                    name="OHLC",
                )
            )

            # Добавление аномальной точки
            anomaly_point = segment[segment["Date"] == idx_date]
            if not anomaly_point.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_point["Date"],
                        y=anomaly_point["Close"],
                        mode="markers",
                        marker=dict(size=10, color="red"),
                        name="Аномалия",
                    )
                )

            # Настройки графика
            fig.update_layout(
                title=f"Аномалия {i + 1} ({method})",
                xaxis_title="Дата",
                yaxis_title="Цена",
                xaxis_rangeslider_visible=False,
                template="plotly_white",
            )

            # Сохранение графика
            file_name = output_dir / f"{method}_anomaly_{i + 1}.html"
            try:
                fig.write_html(str(file_name))
                logger.info(f"Сохранено: {file_name}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении графика для {method}, аномалия {i + 1}: {e}")


# Анализ выбросов
def analyze_outliers(df: pd.DataFrame, analysis_result: dict) -> pd.DataFrame:
    """
    Анализ выбросов: IQR, Z-оценка, Local Outlier Factor, Isolation Forest.
    """
    logger.info("Анализ выбросов...")
    # IQR
    q1 = df[["Open", "High", "Low", "Close"]].quantile(0.25)
    q3 = df[["Open", "High", "Low", "Close"]].quantile(0.75)
    iqr = q3 - q1
    outliers_iqr = (df[["Open", "High", "Low", "Close"]] < (q1 - 1.5 * iqr)) | (
        df[["Open", "High", "Low", "Close"]] > (q3 + 1.5 * iqr)
    )
    df["Outlier_IQR"] = outliers_iqr.any(axis=1)
    analysis_result["Outlier_IQR"] = int(df["Outlier_IQR"].sum())
    logger.info(f"IQR выбросы обнаружены: {df['Outlier_IQR'].sum()}")

    # Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20)
    df["Outlier_LOF"] = lof.fit_predict(df[["Open", "High", "Low", "Close", "Volume"]]) == -1
    analysis_result["Outlier_LOF"] = int(df["Outlier_LOF"].sum())
    logger.info(f"LOF выбросы обнаружены: {df['Outlier_LOF'].sum()}")

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    df["Outlier_IsolationForest"] = (
        iso_forest.fit_predict(df[["Open", "High", "Low", "Close", "Volume"]]) == -1
    )
    analysis_result["Outlier_IsolationForest"] = int(df["Outlier_IsolationForest"].sum())
    logger.info(f"Isolation Forest выбросы обнаружены: {df['Outlier_IsolationForest'].sum()}")

    return analysis_result


# Функция для проверки данных на пропуски и ошибки
def check_missing_and_errors(df: pd.DataFrame, analysis_result: dict) -> pd.DataFrame:
    """
    Проверяет пропуски и ошибки в данных.
    """
    logger.info("Проверка пропусков...")
    missing_info = df.isnull().sum()
    if missing_info.any():
        analysis_result["missing_values"] = True
        logger.warning(f"Найдены пропуски:\n{missing_info}")
    else:
        analysis_result["missing_values"] = False
        logger.success("Пропуски не найдены.")

    logger.info("Проверка логических ошибок...")
    error_rows = df[
        (df["Low"] > df["Open"])
        | (df["Open"] > df["High"])
        | (df["Low"] > df["Close"])
        | (df["Close"] > df["High"])
        | (df["Volume"] < 0)
    ]
    if not error_rows.empty:
        analysis_result["logic_errors"] = True
        logger.warning(f"Найдены логические ошибки:\n{error_rows}")
    else:
        analysis_result["logic_errors"] = False
        logger.success("Логических ошибок не найдено.")

    logger.info("Проверка временных интервалов...")
    df["TimeDiff"] = df["Date"].diff()
    irregular_intervals = df[df["TimeDiff"].dt.days > 4]
    logger.info(f"Max GAP: {int(df['TimeDiff'].dt.days.max())}")
    if not irregular_intervals.empty:
        analysis_result["interval_errors"] = True
        logger.warning(f"Обнаружены нерегулярные интервалы: {len(irregular_intervals)}")
    else:
        analysis_result["interval_errors"] = False
        logger.success("Временные интервалы равномерны.")
    analysis_result["max_gap_day"] = int(df["TimeDiff"].dt.days.max())
    return analysis_result


# Функция анализа данных
def check_data_quality(file_path: Path, figures_dir: Path) -> dict:
    """
    Анализирует данные на пропуски и выбросы, создает графики с японскими свечами.

    Параметры:
        file_path (Path): Путь к файлу данных.
        figures_dir (Path): Директория для сохранения графиков.

    Возвращает:
        dict: Результаты анализа.
    """
    analysis_result = {"file": str(file_path.name)}  # Сохраняем имя файла
    try:
        data = pd.read_csv(file_path)
        data["Date"] = pd.to_datetime(data["Date"])
    except Exception as e:
        logger.warning(f"Ошибка загрузки файла {file_path}: {e}")
        analysis_result["error"] = str(e)
        return analysis_result

    column_to_check = "Close" if "Close" in data.columns else None
    if not column_to_check:
        logger.warning(f"Столбец 'Close' отсутствует в файле {file_path}. Пропущено.")
        analysis_result["status"] = "skipped"
        return analysis_result

    # Проверка на пропуски и ошибки
    check_missing_and_errors(df=data, analysis_result=analysis_result)
    # Анализ выбросов
    analyze_outliers(df=data, analysis_result=analysis_result)
    # Сохранение аномальных отрезков
    save_anomalies_visual_candlestick(df=data, output_dir=figures_dir)

    logger.info(f"analysis_result: {analysis_result}")

    return analysis_result


# Основная функция анализа качества данных
def analyze_data_quality(
    input_dir: Path, report_file: Path, figures_dir: Path, top_5: bool = False
):
    """
    Выполняет анализ качества данных.

    Параметры:
        input_dir (Path): Путь к директории с файлами данных.
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
            tick_name = str(file_path).split("/")[-1].split(".")[0]
            result = check_data_quality(file_path, figures_dir / tick_name)
            report.append(result)
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {file_path}: {e}")

    save_html_report(report, report_file)


# Основная команда CLI
@app.command()
def main(
    sp500_input_dir: Path = RAW_DATA_DIR / "sp500_data",
    crypto_input_dir: Path = RAW_DATA_DIR / "crypto_data",
    sp500_report_file: Path = REPORTS_DIR / "sp500_report.html",
    crypto_report_file: Path = REPORTS_DIR / "crypto_report.html",
    sp500_figures_dir: Path = FIGURES_DIR / "sp500",
    crypto_figures_dir: Path = FIGURES_DIR / "crypto",
    top_5: bool = typer.Option(True, help="Обработать только топ-5 тикеров S&P 500."),
):
    """
    Главная функция для анализа качества данных.

    Параметры:
        sp500_input_dir (Path): Директория данных S&P 500.
        crypto_input_dir (Path): Директория данных криптовалют.
        sp500_report_file (Path): Путь для сохранения отчета по S&P 500.
        crypto_report_file (Path): Путь для сохранения отчета по криптовалютам.
        sp500_figures_dir (Path): Директория для сохранения графиков S&P 500.
        crypto_figures_dir (Path): Директория для сохранения графиков криптовалют.
        top_5 (bool): Обработать только топ-5 тикеров.
    """
    analyze_data_quality(sp500_input_dir, sp500_report_file, sp500_figures_dir, top_5)
    analyze_data_quality(crypto_input_dir, crypto_report_file, crypto_figures_dir)

    logger.success("Анализ качества данных завершен.")


if __name__ == "__main__":
    app()
