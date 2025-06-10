import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

# Model paths
_model_dir = os.path.join(os.path.dirname(__file__), "models")
_prototxt_faces = os.path.join(_model_dir, "deploy_faces.prototxt")
_caffemodel_faces = os.path.join(_model_dir, "faces_dnn.caffemodel")
_prototxt_person = os.path.join(_model_dir, "deploy_person.prototxt.txt")
_caffemodel_person = os.path.join(_model_dir, "person_dnn.caffemodel")

# Load networks once at module load
_face_net = cv2.dnn.readNetFromCaffe(_prototxt_faces, _caffemodel_faces)
_person_net = cv2.dnn.readNetFromCaffe(_prototxt_person, _caffemodel_person)


def correct_image_orientation(image_path: str, save_path: Optional[str] = None) -> None:
    """
    Auto-correct image orientation based on face/person detection scores.
    
    Args:
        image_path: Path to input image file.
        save_path: Optional path to save corrected image. If not provided, overwrite original.
    """
    angle = _detect_landscape_orientation(image_path)
    print(f"{image_path}: rotating {angle}°" if angle else f"{image_path}: no rotation")

    if angle == 0:
        return

    img = Image.open(image_path)
    rotated = img.rotate(-angle, expand=True)

    output_path = save_path or image_path
    rotated.save(output_path, quality=100)


def visualize_detections(image_path: str, mode: str = "face", conf_threshold: float = 0.7) -> None:
    """
    Draw DNN detection boxes on original and 90-degree rotated image for debugging.
    
    Args:
        image_path: Path to input image file.
        mode: Detection mode: 'face' or 'person'.
        conf_threshold: Minimum confidence threshold for detection.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found:", image_path)
        return

    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    try:
        detect_fn = {
            "face": _detect_faces,
            "person": _detect_persons,
        }[mode]
    except KeyError:
        raise ValueError("Mode must be 'face' or 'person'")

    boxes_0 = detect_fn(image, conf_threshold)
    boxes_90 = detect_fn(rotated_90, conf_threshold)

    for box in boxes_0:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for box in boxes_90:
        x, y, w, h = box
        cv2.rectangle(rotated_90, (x, y), (x + w, y + h), (0, 0, 255), 2)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.dirname(image_path)
    suffix = "dnn" if mode == "face" else "person_dnn"
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_0_{suffix}.jpg"), image)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_90_{suffix}.jpg"), rotated_90)

    print(f"{image_path}: DNN {mode}s - 0°: {len(boxes_0)}, 90°: {len(boxes_90)}")


# --- Internal Functions ---

def _detect_landscape_orientation(image_path: str) -> int:
    """Detect whether a 90-degree rotation improves orientation using DNN detections."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return 0

    height, width = image.shape[:2]
    if width > height:
        print(f"{image_path}: Already landscape, skipping rotation")
        return 0

    rotated = _rotate_cv_image(image, 90)

    # Count faces and persons in both orientations
    faces_0 = _count_faces(image, conf_threshold=0.7)
    faces_90 = _count_faces(rotated, conf_threshold=0.7)

    persons_0 = _count_persons(image, conf_threshold=0.5)
    persons_90 = _count_persons(rotated, conf_threshold=0.5)

    print(
        f"{image_path}: faces_0={faces_0}, faces_90={faces_90}, "
        f"persons_0={persons_0}, persons_90={persons_90}"
    )

    score_0 = faces_0 + persons_0
    score_90 = faces_90 + persons_90

    return 90 if score_90 > score_0 else 0


def _rotate_cv_image(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by specified angle (only supports 90°)."""
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def _count_faces(image: np.ndarray, *, conf_threshold: float = 0.7) -> int:
    """Count faces in image using DNN model."""
    return len(_detect_faces(image, conf_threshold))


def _count_persons(image: np.ndarray, *, conf_threshold: float = 0.5) -> int:
    """Count persons in image using DNN model."""
    return len(_detect_persons(image, conf_threshold))


def _detect_faces(image: np.ndarray, conf_threshold: float = 0.7) -> List[Tuple[int, int, int, int]]:
    """Detect faces in image using DNN model and return bounding boxes."""
    return _run_dnn_detection(
        image=image,
        net=_face_net,
        target_class_id=None,
        conf_threshold=conf_threshold,
        mean=(104.0, 177.0, 123.0),
        scalefactor=1.0,
    )


def _detect_persons(image: np.ndarray, conf_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """Detect persons in image using DNN model and return bounding boxes."""
    return _run_dnn_detection(
        image=image,
        net=_person_net,
        target_class_id=15,
        conf_threshold=conf_threshold,
        mean=127.5,
        scalefactor=0.007843,
    )


def _run_dnn_detection(
    image: np.ndarray,
    net,
    *,
    target_class_id: Optional[int],
    conf_threshold: float,
    mean,
    scalefactor: float,
) -> List[Tuple[int, int, int, int]]:
    """
    Run DNN object detection and return filtered bounding boxes.
    
    Args:
        image: Input image.
        net: DNN network model.
        target_class_id: Only keep this class ID (if provided).
        conf_threshold: Confidence threshold.
        mean: Mean values for normalization.
        scalefactor: Scale factor for blob conversion.
        
    Returns:
        List of bounding boxes in (x, y, width, height) format.
    """
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), scalefactor, (300, 300), mean)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < conf_threshold:
            continue

        if target_class_id is not None:
            class_id = int(detections[0, 0, i, 1])
            if class_id != target_class_id:
                continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes
