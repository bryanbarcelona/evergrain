from typing import List, Optional, Tuple

import cv2
import numpy as np

from evergrain.core.orientation.model_loader import load_face_net, load_person_net


def detect_faces(image: np.ndarray, *, conf_threshold: float = 0.7) -> List[Tuple[int, int, int, int]]:
    return _run_dnn_detection(
        image=image,
        net=load_face_net(),
        target_class_id=None,
        conf_threshold=conf_threshold,
        mean=(104.0, 177.0, 123.0),
        scalefactor=1.0,
    )


def detect_persons(image: np.ndarray, *, conf_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    return _run_dnn_detection(
        image=image,
        net=load_person_net(),
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
            if int(detections[0, 0, i, 1]) != target_class_id:
                continue
        x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype('int')
        boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes
