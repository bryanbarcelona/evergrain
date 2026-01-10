import cv2
import numpy as np

from evergrain.core.orientation.detector import detect_faces, detect_persons


def detect_landscape_orientation(image_path: str) -> int:
    image = cv2.imread(image_path)
    if image is None:
        print(f'Could not load image: {image_path}')
        return 0

    height, width = image.shape[:2]
    if width > height:
        print(f'{image_path}: Already landscape, skipping rotation')
        return 0

    rotated = rotate_cv_image(image, 90)

    faces_0 = len(detect_faces(image, conf_threshold=0.7))
    faces_90 = len(detect_faces(rotated, conf_threshold=0.7))
    persons_0 = len(detect_persons(image, conf_threshold=0.5))
    persons_90 = len(detect_persons(rotated, conf_threshold=0.5))

    print(f'{image_path}: faces_0={faces_0}, faces_90={faces_90}, persons_0={persons_0}, persons_90={persons_90}')

    return 90 if (faces_90 + persons_90) > (faces_0 + persons_0) else 0


def rotate_cv_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image
