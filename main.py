from PIL import Image as imgr
import glob
from cropper import PhotoSplitter, ScanBackground
from orientation import auto_correct_photo_orientation, draw_detected_faces, draw_faces_debug_dnn
import os
import logging

# Paths and parameters
background_path = r"D:\Coding\evergrain\configs\background_reference.jpg"
scans_dir = r"C:\Users\bryan\Desktop\scanner_test"
output_dir = r"C:\Users\bryan\Desktop\output"
dpi = 600
contrast = 15.0
shrink = 15
deskew = True

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

try:
    # Load background image
    with imgr.open(background_path) as blank_img:
        background = ScanBackground.from_image(blank_img, dpi=dpi)
        logging.info(f"Loaded background: median={background.median_color}, std={background.color_variation}")

    # Find scan images
    scanned_raw_image_list = glob.glob(os.path.join(scans_dir, "*.jpg")) + \
                            glob.glob(os.path.join(scans_dir, "*.png"))
    if not scanned_raw_image_list:
        logging.error(f"No scan images found in {scans_dir}")
        exit(1)

    # Process each scan
    for i, scan_path in enumerate(scanned_raw_image_list):
        logging.info(f"Processing scan: {scan_path}")
        try:
            with imgr.open(scan_path) as scan_img:
                scan = PhotoSplitter(
                    scan_img,
                    background,
                    dpi=dpi,
                    contrast=contrast,
                    shrink=shrink,
                    deskew=deskew
                )
                for index, photo in enumerate(scan):
                    output_path = os.path.join(
                        output_dir,
                        f"{str(i).zfill(3)}image-{index}.jpg"
                    )
                    photo.save(output_path, dpi=(dpi, dpi), quality=90)
                    logging.info(f"Saved photo: {output_path}")
        except Exception as e:
            logging.error(f"Error processing {scan_path}: {e}")
except Exception as e:
    logging.error(f"Error loading background {background_path}: {e}")

for filename in os.listdir(output_dir):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(output_dir, filename)
        draw_faces_debug_dnn(image_path)
        draw_detected_faces(image_path, output_folder=output_dir)
        auto_correct_photo_orientation(image_path)