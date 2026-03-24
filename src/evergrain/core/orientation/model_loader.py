from functools import lru_cache
from pathlib import Path

import cv2


def _model_path(filename: str) -> str:
    """Absolute path to a file inside THIS module's models/ directory."""
    return str(Path(__file__).with_suffix('').parent / 'models' / filename)


@lru_cache(maxsize=1)
def load_face_net() -> cv2.dnn.Net:
    """Load the face detection Caffe model.

    Returns:
        cv2.dnn.Net: The loaded face detection network.
    """
    prototxt = _model_path('deploy_faces.prototxt')
    caffemodel = _model_path('faces_dnn.caffemodel')
    return cv2.dnn.readNetFromCaffe(prototxt, caffemodel)


@lru_cache(maxsize=1)
def load_person_net() -> cv2.dnn.Net:
    """Load the person detection Caffe model.

    Returns:
        cv2.dnn.Net: The loaded person detection network.
    """
    prototxt = _model_path('deploy_person.prototxt.txt')
    caffemodel = _model_path('person_dnn.caffemodel')
    return cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
