import cv2
import numpy as np

from evergrain.core.orientation.model_loader import load_face_net, load_person_net


def detect_faces(image: np.ndarray, conf_threshold: float = 0.7) -> list[tuple[int, int, int, int]]:
    """Detect faces in an image using the face detection model.

    Args:
        image: Input image as numpy array (BGR format)
        conf_threshold: Confidence threshold for detection

    Returns:
        List of bounding boxes as (x, y, width, height)
    """
    return _run_dnn_detection(
        image=image,
        net=load_face_net(),
        target_class_id=None,
        conf_threshold=conf_threshold,
        mean=(104.0, 177.0, 123.0),  # Face model specific means per channel (B, G, R)
        scalefactor=1.0,
    )


def detect_persons(image: np.ndarray, conf_threshold: float = 0.5) -> list[tuple[int, int, int, int]]:
    """Detect persons in an image using the person detection model.

    Args:
        image: Input image as numpy array (BGR format)
        conf_threshold: Confidence threshold for detection

    Returns:
        List of bounding boxes as (x, y, width, height)
    """
    return _run_dnn_detection(
        image=image,
        net=load_person_net(),
        target_class_id=15,  # COCO class ID for person
        conf_threshold=conf_threshold,
        mean=127.5,
        scalefactor=0.007843,
    )


def _run_dnn_detection(
    image: np.ndarray,
    net: cv2.dnn.Net,
    *,
    target_class_id: int | None,
    conf_threshold: float,
    mean: float | tuple[float, float, float],
    scalefactor: float,
) -> list[tuple[int, int, int, int]]:
    """Run DNN detection on an image.

    Args:
        image: Input image as numpy array
        net: Loaded DNN model
        target_class_id: Optional class ID to filter detections
        conf_threshold: Confidence threshold
        mean: Mean subtraction values (scalar or tuple)
        scalefactor: Scale factor for pixel values

    Returns:
        List of bounding boxes as (x, y, width, height)
    """
    h, w = image.shape[:2]

    # Create blob from image
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=scalefactor,
        size=(300, 300),
        mean=mean,
    )  # ty: ignore[no-matching-overload]

    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter by confidence and class ID
        if confidence >= conf_threshold and (target_class_id is None or int(detections[0, 0, i, 1]) == target_class_id):
            # Convert from (x1, y1, x2, y2) normalized to absolute coordinates
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype('int')
            # Convert to (x, y, width, height) format
            boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes
