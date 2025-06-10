from PIL import Image as imgr
import glob
from cropper import PhotoSplitter, ScanBackground
from orientation import correct_image_orientation, visualize_detections
from enhancements import enhance_image_quality
import os
import logging
import shutil

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

def reset_testing_folder(originals_folder: str, testing_folder: str) -> None:
    """
    Deletes the content of the testing folder and copies all contents
    from the originals folder into the testing folder.

    Args:
        originals_folder: Path to the folder containing your safe original images.
        testing_folder: Path to the folder where you want to perform testing.
    """
    if not os.path.exists(originals_folder):
        raise FileNotFoundError(f"Originals folder not found: {originals_folder}")

    # 1. Delete the content of the testing folder
    if os.path.exists(testing_folder):
        print(f"Deleting contents of: {testing_folder}...")
        for item in os.listdir(testing_folder):
            item_path = os.path.join(testing_folder, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove file or link
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path) # Remove directory and its contents
        print("Testing folder cleared.")
    else:
        # If the testing folder doesn't exist, create it
        os.makedirs(testing_folder)
        print(f"Testing folder created: {testing_folder}")

    # 2. Copy everything from the safe originals to the now empty testing folder
    print(f"Copying files from '{originals_folder}' to '{testing_folder}'...")
    for item in os.listdir(originals_folder):
        s = os.path.join(originals_folder, item)
        d = os.path.join(testing_folder, item)
        if os.path.isfile(s):
            shutil.copy2(s, d) # copy2 preserves metadata
        elif os.path.isdir(s):
            shutil.copytree(s, d) # copytree copies entire directories
    print("Test environment reset successfully!")

# original_dir = r"c:\Users\bryan\Desktop\output - Copy"
# test_dir = r"c:\Users\bryan\Desktop\output"
# reset_testing_folder(original_dir, test_dir)

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
                    photo.save(output_path, dpi=(dpi, dpi), quality=100)
                    logging.info(f"Saved photo: {output_path}")
        except Exception as e:
            logging.error(f"Error processing {scan_path}: {e}")
except Exception as e:
    logging.error(f"Error loading background {background_path}: {e}")



for filename in os.listdir(output_dir):
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(output_dir, filename)
        # visualize_detections(image_path, mode="face", conf_threshold=0.7)
        # visualize_detections(image_path, mode="person", conf_threshold=0.7)
        correct_image_orientation(image_path)
        enhance_image_quality(image_path)
