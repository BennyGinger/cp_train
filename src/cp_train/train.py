from pathlib import Path
import logging

from cellpose import io, models, train

from cp_train.model import TrainSettings


logger = logging.getLogger(__name__)

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
    
    logger.debug(f"Final training loss: {train_losses[-1]:.4f}")
    if test_losses is not None:
        logger.debug(f"Final test loss: {test_losses[-1]:.4f}")
        if train_losses[-1] > test_losses[0]:
            logger.warning("Warning: Final training loss is higher than initial test loss, which may indicate overfitting.")
    return Path(model_path)