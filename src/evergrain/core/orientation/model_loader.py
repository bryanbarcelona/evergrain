from pathlib import Path
from typing import Any

import cv2


def _model_path(filename: str) -> str:
    """Absolute path to a file inside THIS module's models/ directory."""
    return str(Path(__file__).with_suffix('').parent / 'models' / filename)


def load_face_net() -> Any:
    if not hasattr(load_face_net, '_net'):
        prototxt = _model_path('deploy_faces.prototxt')
        caffemodel = _model_path('faces_dnn.caffemodel')
        load_face_net._net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    return load_face_net._net


def load_person_net() -> Any:
    if not hasattr(load_person_net, '_net'):
        prototxt = _model_path('deploy_person.prototxt.txt')
        caffemodel = _model_path('person_dnn.caffemodel')
        load_person_net._net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    return load_person_net._net
