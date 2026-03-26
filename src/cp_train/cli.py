from pathlib import Path
import typer

from cp_train.api import train_dataset


app = typer.Typer(
    help="Train a CP3 Cellpose model from CP4-generated annotations."
)


@app.command()
def train(
    root_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Dataset root containing 'train/' and optional 'test/'.",),
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to TOML config file.",),
) -> None:
    """
    Run the full workflow:
    1. CP4 annotation through cp_runner
    2. CP3 training in cp_train
    """
    typer.echo(f"Dataset: {root_dir}")
    typer.echo(f"Config:  {config}")
    
    try:
        model_path = train_dataset(root_dir=root_dir, config_path=config)
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho(f"Training complete.", fg=typer.colors.GREEN)
    typer.echo(f"Model saved at: {model_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()