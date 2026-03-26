from pathlib import Path
import logging

from cellpose import io

from cp_runner.model_cache import segment_model_cache
from cp_runner.filesystem import find_tif_files, load_config
from cp_runner.array import load_image, validate_array


logger = logging.getLogger(__name__)


# ------ Public API --------
def annotate_dataset(root_dir: Path | str, config_path: str | Path) -> None:
    
    annot = load_config(config_path)
    
    files = find_tif_files(root_dir)
    if not files:
        logger.warning(f"No TIFF files found in {root_dir}")
        return
    
    model_settings = {}
    segment_settings = {**annot.__dict__}
    
    requires_Zaxis = False
    if annot.do_3D or annot.stitch_threshold > 0.0:
        requires_Zaxis = True
    
    model_settings['user_settings'] = segment_settings
    wrapper = segment_model_cache.get_wrapper(model_settings)
    
    for file in files:
        logger.debug(f"Processing file: {file}")
        
        img, axes = load_image(file)
        validate_array(img, axes, requires_Zaxis)
        
        wrapper.run(img, axes)
        results = wrapper.segmentation_result
        if results is None:
            raise RuntimeError(f"Segmentation failed for {file} with settings {segment_settings}")
        
        io.masks_flows_to_seg(images=[img],
                            masks=results.masks[0],
                            flows=results.flows[0],
                            file_names=[file.as_posix()])
    
    logger.info(f"Annotation complete for dataset at {root_dir} with config {config_path}")
            


if __name__ == "__main__":
    
    dir = Path('/media/ben/Analysis/Python/Images/tiff/Run4/im_seq')
    config = Path('/media/ben/Analysis/Python/Repos/cp_train/src/cp_train/training_settings.toml')
    
    annotate_dataset(dir, config)