from pathlib import Path

from cellpose import io

from cp_runner.model_cache import segment_model_cache
from cp_runner.filesystem import find_tif_files, load_config
from cp_runner.array import load_image, validate_array


# ------ Public API --------
def annotate_dataset(root_dir: Path | str, config_path: str | Path) -> None:
    
    annot = load_config(config_path)
    
    files = find_tif_files(root_dir)
    if not files:
        print(f"No TIFF files found in {root_dir}")
        return
    
    model_settings = {}
    segment_settings = {**annot.__dict__}
    
    requires_Zaxis = False
    if annot.do_3D or annot.stitch_threshold > 0.0:
        requires_Zaxis = True
    
    model_settings['user_settings'] = segment_settings
    wrapper = segment_model_cache.get_wrapper(model_settings)
    
    for file in files:
        
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
    
    print(f"Annotation complete for dataset at {root_dir} with config {config_path}")
            


if __name__ == "__main__":
    
    dir = Path('/home/ben/EBlabDrive/Imaging/Ben/training/dataset')
    config = Path('/home/ben/EBlabDrive/Imaging/Ben/training/dataset/training_settings.toml')
    
    annotate_dataset(dir, config)