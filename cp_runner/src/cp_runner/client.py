from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

from tifffile import TiffFile, imread
from numpy.typing import NDArray
from cellpose import io
from progress_bar import pbar

from cp_runner.model_cache import segment_model_cache


logger = logging.getLogger(__name__)


# ------ Public API --------
def annotate_dataset(root_dir: Path | str, pretrained_model: str = "cpsam", diameter=None, do_3D=False, stitch_threshold=0.0, flow_threshold=0.4, cellprob_threshold=0.0, **kwargs,) -> None:
    
    files = _find_tif_files(root_dir)
    if not files:
        logger.warning(f"No TIFF files found in {root_dir}")
        return
    
    model_settings = {}
    segment_settings = {
        'pretrained_model': pretrained_model,
        'diameter': diameter,
        'do_3D': do_3D,
        'stitch_threshold': stitch_threshold,
        'flow_threshold': flow_threshold,
        'cellprob_threshold': cellprob_threshold,
    }
    segment_settings.update(kwargs)
    requires_Zaxis = False
    if segment_settings['do_3D'] or segment_settings['stitch_threshold'] > 0.0:
        requires_Zaxis = True
    
    model_settings['user_settings'] = segment_settings
    wrapper = segment_model_cache.get_wrapper(model_settings)
    
    with pbar(total=len(files), desc="Annotating dataset", logs="off") as pb:
        for file in files:
            logger.debug(f"Processing file: {file}")
            
            img, axes = _load_image(file)
            _validate_array(img, axes, requires_Zaxis)
            
            wrapper.run(img, axes)
            results = wrapper.segmentation_result
            if results is None:
                raise RuntimeError(f"Segmentation failed for {file} with settings {segment_settings}")
            
            io.masks_flows_to_seg(images=[img],
                                masks=results.masks[0],
                                flows=results.flows[0],
                                file_names=[file.as_posix()])
            pb.advance()


# ------ Internal utilities --------
def _find_tif_files(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir)
    if root.is_file():
        return [root] if root.suffix.lower() in {".tif", ".tiff"} else []

    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})


def _load_image(img_path: Path) -> tuple[NDArray[Any], str]:
    """
    Load an image from the given path using tifffile. This function will be fed real exisiting path from _find_tif_files, so we can assume the file exists and is a valid TIFF file.
    """
    axe = _get_axis_labels(img_path)
    return imread(img_path), axe


def _get_axis_labels(img_path: Path) -> str:
    """
    Get the axis labels from the TIFF file. This function assumes that all series in the TIFF file have the same axes.
    """
    with TiffFile(img_path) as tif:
        _axes = [s.axes for s in tif.series]
    return _axes[0]


def _validate_array(arr: NDArray[Any], axes: str, requires_Zaxis: bool) -> None:
    
    if len(axes) != arr.ndim:
        raise ValueError(f"Number of axes in {axes} does not match array dimensions ({arr.ndim})")
    
    if 'T' in axes:
        raise ValueError(f"Only single timepoint is supported, but found 'T' axis in {axes}")
    
    if requires_Zaxis and 'Z' not in axes:
        raise ValueError(f"3D annotation requested, but no 'Z' axis found in {axes}")
    
    if 'C' in axes and arr.shape[axes.index('C')] > 3:
        raise ValueError(f"Too many channels ({arr.shape[axes.index('C')]}) found in {axes}, maximum supported is 3")


if __name__ == "__main__":
    
    dir = Path('/media/ben/Analysis/Python/Images/tiff/Run4/im_seq')
    
    annotate_dataset(dir)