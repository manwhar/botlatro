from dataclasses import dataclass


@dataclass
class Template:
    name: str
    filenames: list[str]
    rois: list[tuple[float, float, float, float]]
