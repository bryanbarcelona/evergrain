from pathlib import Path
from typing import Literal

import cv2
from PIL import Image

from evergrain.exceptions.enhancement import InvalidImageError
from evergrain.utils.atomic import atomic_replace


def load_image(path: Path | str) -> Image.Image:
    path = Path(path)
    cv_img = cv2.imread(str(path))
    if cv_img is None:
        raise InvalidImageError(f'OpenCV cannot read file: {path}')
    rgb = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    if rgb.mode != 'RGB':
        rgb = rgb.convert('RGB')

    return rgb


def save_image(image: Image.Image, path: Path | str, *, file_format: str | None = None) -> None:
    path = Path(path)
    if file_format is None:
        file_format = _guess_format(path)
    image.save(path, format=file_format, quality=100)


def overwrite_image(image: Image.Image, target: Path | str) -> None:
    target = Path(target)
    atomic_replace(target, lambda tmp: image.save(tmp, format=_guess_format(target), quality=100))


def _guess_format(path: Path) -> Literal['JPEG', 'PNG', 'TIFF']:
    suffix = path.suffix.lower().lstrip('.')
    formats: dict[str, Literal['JPEG', 'PNG', 'TIFF']] = {
        'jpg': 'JPEG',
        'jpeg': 'JPEG',
        'png': 'PNG',
        'tif': 'TIFF',
        'tiff': 'TIFF',
    }
    return formats.get(suffix, 'PNG')
