from pathlib import Path

from cellpose import io, models, train
import numpy as np

from cp_train.model import TrainSettings


def run_training(train_dir: str | Path, test_dir: str | Path | None, train_settings: TrainSettings) -> Path:
    
    output = io.load_train_test_data(
        train_dir=Path(train_dir).resolve().as_posix(),
        test_dir=Path(test_dir).resolve().as_posix() if test_dir else None,
        image_filter=None,
        mask_filter="_seg.npy",
        look_one_level_down=False,)

    images, labels, _, test_images, test_labels, _ = output

    model = models.CellposeModel(
        gpu=True,
        model_type=train_settings.pretrained_model,)

    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        test_data=test_images,
        test_labels=test_labels,
        channels=[0,0], #type: ignore
        n_epochs=train_settings.n_epochs,
        model_name=train_settings.model_name,
        SGD=train_settings.SGD,
        learning_rate=train_settings.learning_rate,
        weight_decay=train_settings.weight_decay,
        save_path=Path(train_dir).parent.as_posix(),)
    
    print(f"Train loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")

    if test_losses is not None and len(test_losses) > 0:
        print(f"Test loss:  {test_losses[0]:.4f} → {test_losses[-1]:.4f}")
        best_epoch = int(np.argmin(test_losses))
        print(f"Best test loss at epoch {best_epoch}: {test_losses[best_epoch]:.4f}")
        
        if test_losses[-1] > min(test_losses):
            print("Test loss worsened after best point → possible overfitting.")
    return Path(model_path)