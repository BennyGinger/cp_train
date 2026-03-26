from pathlib import Path
from typing import Any

from numpy.typing import NDArray
from tifffile import TiffFile, imread



# ----- Public APIs --------
def load_image(img_path: Path) -> tuple[NDArray[Any], str]:
    """
    Load an image from the given path using tifffile. This function will be fed real exisiting path from _find_tif_files, so we can assume the file exists and is a valid TIFF file.
    """
    axe = _get_axis_labels(img_path)
    return imread(img_path), axe


def validate_array(arr: NDArray[Any], axes: str, requires_Zaxis: bool) -> None:
    
    if len(axes) != arr.ndim:
        raise ValueError(f"Number of axes in {axes} does not match array dimensions ({arr.ndim})")
    
    if 'T' in axes:
        raise ValueError(f"Only single timepoint is supported, but found 'T' axis in {axes}")
    
    if requires_Zaxis and 'Z' not in axes:
        raise ValueError(f"3D annotation requested, but no 'Z' axis found in {axes}")
    
    if 'C' in axes and arr.shape[axes.index('C')] > 3:
        raise ValueError(f"Too many channels ({arr.shape[axes.index('C')]}) found in {axes}, maximum supported is 3")
   

# ------ Internal functions --------
def _get_axis_labels(img_path: Path) -> str:
    """
    Get the axis labels from the TIFF file. This function assumes that all series in the TIFF file have the same axes.
    """
    with TiffFile(img_path) as tif:
        _axes = [s.axes for s in tif.series]
    return _axes[0]   