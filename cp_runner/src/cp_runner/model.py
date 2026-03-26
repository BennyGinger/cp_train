from dataclasses import dataclass


@dataclass
class AnnotationSettings:
    pretrained_model: str = "cpsam"
    diameter: float | None = 50
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    do_3D: bool = False
    stitch_threshold: float = 0.0