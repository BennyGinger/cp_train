from pathlib import Path
import tomllib


from cp_runner.model import AnnotationSettings



def load_config(config_path: str | Path) -> AnnotationSettings:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    if "annotation" not in config:
        raise ValueError("Missing [annotation] section in config")

    return AnnotationSettings(**config["annotation"])


def find_tif_files(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir)
    if root.is_file():
        return [root] if root.suffix.lower() in {".tif", ".tiff"} else []

    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})


