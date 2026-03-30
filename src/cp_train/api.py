from pathlib import Path

from cp_train.filesystem import load_config, resolve_train_test_dirs, validate_seg_outputs
from cp_train.cp_runner_invoker import run_cp_runner
from cp_train.train import run_training



def train_dataset(root_dir: str | Path, config_path: str | Path) -> Path:
    
    root_dir = Path(root_dir)
    train_settings = load_config(config_path)
    
    train_dir, test_dir = resolve_train_test_dirs(root_dir)
    
    # annotate training data
    # run_cp_runner(root_dir, Path(config_path))
    
    # Ensure annotation outputs are present for training data
    validate_seg_outputs(root_dir)

    return run_training(train_dir=train_dir,
                        test_dir=test_dir,
                        train_settings=train_settings,)


if __name__ == "__main__":
    
    dir = Path('/home/ben/EBlabDrive/Imaging/Ben/training/dataset')
    config = Path('/home/ben/EBlabDrive/Imaging/Ben/training/dataset/training_settings.toml')
    
    model_path = train_dataset(dir, config)
    print(f"Trained model saved at: {model_path}")