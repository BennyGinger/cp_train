from pathlib import Path
import typer

from cp_runner.client import annotate_dataset

app = typer.Typer(help="Run CP4 annotation for a dataset directory.")


@app.command()
def annotate(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    config: Path = typer.Option(..., "--config", "-c", exists=True, file_okay=True, dir_okay=False),
) -> None:
    
    annotate_dataset(root_dir=input_dir,
                     config_path=config)


def main() -> None:
    app()


if __name__ == "__main__":
    main()