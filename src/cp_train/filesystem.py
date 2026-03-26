from pathlib import Path
import logging
import tomllib

from cp_train.model import TrainSettings


EXPECTED_IMAGE_FILES = {".tif", ".tiff"}
ALLOWED_SUBDIRS = {"train", "test", "models"}

logger = logging.getLogger(__name__)

# ------ Public API --------
def resolve_train_test_dirs(root_dir: Path | str) -> tuple[Path, Path | None]:
    """
    Given a root directory, validate that it contains 'train' and optionally 'test' subdirectories, and return their paths.
    Raises ValueError if the directory structure is not as expected (e.g., if there are TIFF files directly in the root, or if required subdirectories are missing).
    """
    root_dir = Path(root_dir)
    
    # Check if image files are directly in the root directory
    tif_files = sorted(
        p for p in root_dir.glob("*")
        if p.is_file() and p.suffix.lower() in EXPECTED_IMAGE_FILES)
    if tif_files:
        raise ValueError(f"Expected no TIFF files directly in {root_dir}, but found {len(tif_files)}. Please organize your dataset into 'train' and 'test' subdirectories.")
    
    # Validate subdirs
    subdirs = [d for d in root_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"No subdirectories found in {root_dir}. Please organize your dataset into 'train' (compulsory) and 'test' (optional) subdirectories.")
    
    names = {d.name.lower() for d in subdirs}
    unexpected = names - ALLOWED_SUBDIRS
    if unexpected:
        raise ValueError(f"Unexpected subdirectories in {root_dir}: {sorted(unexpected)}. Only 'train' and optional 'test' and 'models' are allowed).")
    
    if "train" not in names:
        raise ValueError(f"No 'train' directory found in {root_dir}.")
    
    train_dir = root_dir / "train"
    if not _contains_image_files(train_dir):
        raise ValueError(f"No {EXPECTED_IMAGE_FILES} files found in training directory {train_dir}. Please ensure your training data is correctly placed.")
    test_dir = (root_dir / "test") if "test" in names else None
    
    if test_dir is not None and not _contains_image_files(test_dir):
        raise ValueError(f"No {EXPECTED_IMAGE_FILES} files found in test directory {test_dir}. Please ensure your test data is correctly placed, or remove the empty 'test' directory.")
    return train_dir, test_dir


def validate_seg_outputs(data_dir: Path) -> None:
    tif_files = sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EXPECTED_IMAGE_FILES)

    missing = []
    for img in tif_files:
        seg = img.with_name(f"{img.stem}_seg.npy")
        if not seg.exists():
            missing.append(seg)

    if missing:
        preview = "\n".join(str(p) for p in missing[:10])
        raise RuntimeError(
            f"Missing {len(missing)} annotation files in {data_dir}. "
            f"First missing:\n{preview}")


def load_config(config_path: str | Path) -> TrainSettings:
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    if "training" not in config:
        raise ValueError("Missing [training] section in config")

    return TrainSettings(**config["training"])


def _contains_image_files(dir: Path) -> bool:
    return any(
        p for p in dir.glob("*")
        if p.is_file() and p.suffix.lower() in EXPECTED_IMAGE_FILES)