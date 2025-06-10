from typing import Optional
from PIL import Image, ImageOps, ImageFilter 
import cv2
import numpy as np


def enhance_image_quality(image_path: str, save_path: Optional[str] = None) -> None:
    """
    Enhance image quality by applying auto tone, auto contrast, auto color correction,
    and despeckling (gentle denoising).

    Args:
        image_path: Path to the input image file.
        save_path: Optional path to save the enhanced image. If not provided, the original file is overwritten.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Convert OpenCV (NumPy array, BGR) to PIL Image (RGB)
    # OpenCV loads as BGR, PIL expects RGB.
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Ensure the image is in RGB mode for consistent processing by PIL functions
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = _auto_tone(image)
    image = _auto_color(image)
    image = _auto_contrast(image)
    image = _despeckle(image)
    image = _sharpen(image)

    output_path = save_path or image_path
    try:
        # Save the enhanced image using PIL
        # PIL will infer the format from the extension.
        image.save(output_path, quality=100)
        print(f"'{image_path}': enhanced and saved to '{output_path}'")
    except Exception as e:
        raise IOError(f"Error saving enhanced image to {output_path}: {e}")
#    cv2.imwrite(output_path, image)
    print(f"{image_path}: enhanced and saved to {output_path}")


# --- Internal Helpers ---


def _auto_tone(image_pil: Image.Image) -> Image.Image:
    """
    Applies auto tone by applying autocontrast to each RGB channel independently.
    Args:
        image_pil: PIL Image object.
    Returns:
        PIL Image object with auto tone applied.
    """
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')

    # Split channels
    r, g, b = image_pil.split()

    # Apply autocontrast to each channel
    r_auto = ImageOps.autocontrast(r, cutoff=0)
    g_auto = ImageOps.autocontrast(g, cutoff=0)
    b_auto = ImageOps.autocontrast(b, cutoff=0)

    # Merge channels back
    return Image.merge("RGB", (r_auto, g_auto, b_auto))

def _auto_contrast(image_pil: Image.Image) -> Image.Image:
    """
    Applies auto contrast to the image.
    Args:
        image_pil: PIL Image object.
    Returns:
        PIL Image object with auto contrast applied.
    """
    if image_pil.mode != 'RGB':
        # AutoContrast generally works better on a grayscale representation or RGB directly.
        # If input is RGBA, it's good practice to convert to RGB first if autocontrast
        # should affect only color channels and not alpha.
        image_pil = image_pil.convert('RGB')

    return ImageOps.autocontrast(image_pil, cutoff=0) # cutoff=0 means no clipping of darkest/lightest pixels

def _auto_color(image_pil: Image.Image,
                black_percentile_cutoff: float = 0.5,  # New parameter for black point
                white_percentile_cutoff: float = 2.0) -> Image.Image: # Updated default from your testing
    """
    Applies auto color and contrast correction by setting per-channel black and white points.
    This aims to maximize dynamic range and neutralize color casts in highlights and shadows.

    Args:
        image_pil: PIL Image object (expected RGB).
        black_percentile_cutoff: The lowest percentile of pixels (e.g., 0.5 means the darkest 0.5%)
                                 to map to pure black (0). Higher values = more aggressive black point.
        white_percentile_cutoff: The highest percentile of pixels (e.g., 2.0 means the brightest 2.0%)
                                 to map to pure white (255). Higher values = more aggressive white point.
    Returns:
        PIL Image object with auto color/contrast applied.
    """
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')

    img_np_rgb = np.array(image_pil, dtype=np.float32)

    corrected_channels = []
    for i in range(3): # Iterate through R, G, B channels
        channel = img_np_rgb[:, :, i]

        # Calculate black point value
        # np.percentile(data, q) gives value BELOW which q% of data falls.
        # So, for the darkest 0.5%, we take the 0.5th percentile.
        low_val = np.percentile(channel.flatten(), black_percentile_cutoff)

        # Calculate white point value
        # For the brightest 2.0%, we take the 98.0th percentile (100 - 2.0).
        high_val = np.percentile(channel.flatten(), 100 - white_percentile_cutoff)

        # Handle cases where the range is flat or invalid to prevent division by zero
        if high_val - low_val <= 0: # If min and max are same or inverted (e.g. all same color or image is weird)
            corrected_channels.append(channel.astype(np.uint8)) # Keep channel as is
            continue

        # Apply levels adjustment: stretch range [low_val, high_val] to [0, 255]
        # new_pixel = (old_pixel - low_val) * (255.0 / (high_val - low_val))
        scaled_channel = (channel - low_val) * (255.0 / (high_val - low_val))

        # Clip values to ensure they stay within 0-255 range and convert to uint8
        corrected_channels.append(np.clip(scaled_channel, 0, 255).astype(np.uint8))

    corrected_img_np = np.stack(corrected_channels, axis=-1)

    return Image.fromarray(corrected_img_np, 'RGB')

# --- Implement Despeckle/Denoising ---
def _despeckle(image_pil: Image.Image) -> Image.Image:
    """
    Applies a Median filter for despeckling/denoising.
    Effective for reducing speckle noise while preserving edges.
    Args:
        image_pil: PIL Image object.
    Returns:
        PIL Image object with denoising applied.
    """
    # A size of 3 (for a 3x3 kernel) is a common starting point for despeckling.
    # Larger sizes remove more noise but also blur more details.
    median_filter_size = 3
    return image_pil.filter(ImageFilter.MedianFilter(size=median_filter_size))

# --- Implement Sharpening ---
def _sharpen(image_pil: Image.Image) -> Image.Image:
    """
    Applies Unsharp Masking for sharpening.
    This method enhances edges and details.
    Args:
        image_pil: PIL Image object.
    Returns:
        PIL Image object with sharpening applied.
    """
    # Parameters for Unsharp Masking (USM)
    # These are common starting points for "auto" sharpening and can be tuned.
    amount_percent = 150  # Strength of sharpening (150% means add 1.5x the edge contrast)
    radius_pixels = 1.0   # How wide the edge detection area is (pixels). Finer details for small radius.
    threshold_contrast = 3 # Minimum contrast required for sharpening to apply (0-255). Helps avoid sharpening noise.

    return image_pil.filter(ImageFilter.UnsharpMask(percent=amount_percent, radius=radius_pixels, threshold=threshold_contrast))

