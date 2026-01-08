import numpy as np
from PIL import Image, ImageOps, ImageFilter


def auto_tone(image: Image.Image) -> Image.Image:
    if image.mode != 'RGB':
        image = image.convert('RGB')

    r, g, b = image.split()
    return Image.merge(
        "RGB",
        (
            ImageOps.autocontrast(r, cutoff=0),
            ImageOps.autocontrast(g, cutoff=0),
            ImageOps.autocontrast(b, cutoff=0),
        ),
    )


def auto_contrast(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return ImageOps.autocontrast(image, cutoff=0)


def auto_color(image: Image.Image, black_pct: float, white_pct: float) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    out_channels = []
    for ch in arr.transpose(2, 0, 1):
        lo = np.percentile(ch, black_pct)
        hi = np.percentile(ch, 100 - white_pct)
        if hi - lo <= 0:
            out_channels.append(ch.astype(np.uint8))
            continue
        scaled = (ch - lo) * (255.0 / (hi - lo))
        out_channels.append(np.clip(scaled, 0, 255).astype(np.uint8))
    return Image.fromarray(np.stack(out_channels, axis=-1), "RGB")


def despeckle(image: Image.Image, size: int) -> Image.Image:
    return image.filter(ImageFilter.MedianFilter(size=size))


def sharpen(image: Image.Image, *, amount: float, radius: float, threshold: int) -> Image.Image:
    return image.filter(ImageFilter.UnsharpMask(percent=int(amount * 100), radius=radius, threshold=threshold))
