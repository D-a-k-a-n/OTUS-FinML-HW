from pathlib import Path
import typer
from visualizer import run_dash_app
from config import INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    data_dir: Path = INTERIM_DATA_DIR,
    top_5: bool = typer.Option(True, help="Set 5 ticks from SP500 (default is True)."),
):
    """
    Запуск веб-приложения для визуализации данных.
    """
    run_dash_app(data_dir=data_dir, top_5=top_5)


if __name__ == "__main__":
    app()
