from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, TYPE_CHECKING

from evergrain.core.enhancement.params import EnhancementParams
from evergrain.core.enhancement import io, filters
from evergrain.core.exceptions.enhancement import InvalidImageError

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class EnhancementEngine:
    def __init__(self, params: EnhancementParams | None = None) -> None:
        self.params = params or EnhancementParams()

    def enhance(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        p = self.params
        if p.auto_tone:
            image = filters.auto_tone(image)
        if p.auto_color:
            image = filters.auto_color(image, p.black_percentile, p.white_percentile)
        if p.auto_contrast:
            image = filters.auto_contrast(image)
        if p.despeckle_size > 0:
            image = filters.despeckle(image, p.despeckle_size)
        if p.usm_amount > 0:
            image = filters.sharpen(
                image,
                amount=p.usm_amount,
                radius=p.usm_radius,
                threshold=p.usm_threshold,
            )
        return image

    def enhance_file(
        self,
        in_path: Path | str,
        out_path: Path | str | None = None,
        *,
        format: Literal["JPEG", "PNG", "TIFF"] | None = None,
    ) -> None:
        in_path = Path(in_path)
        if not in_path.exists():
            raise InvalidImageError(f"Input file not found: {in_path}")

        image = io.load_image(in_path)
        enhanced = self.enhance(image)

        if out_path is None:
            io.overwrite_image(enhanced, in_path)
            logger.debug("Enhanced image saved in-place: %s", in_path)
        else:
            io.save_image(enhanced, out_path, format=format)
            logger.debug("Enhanced image saved to: %s", out_path)
