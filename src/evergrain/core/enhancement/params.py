from dataclasses import dataclass


@dataclass(slots=True)
class EnhancementParams:
    auto_tone: bool = True
    auto_color: bool = True
    auto_contrast: bool = True
    black_percentile: float = 0.5
    white_percentile: float = 2.0
    despeckle_size: int = 3
    usm_amount: float = 1.5
    usm_radius: float = 1.0
    usm_threshold: int = 3
