from pathlib import Path

import cv2

from evergrain.core.orientation.detector import detect_faces, detect_persons
from evergrain.core.orientation.rotator import detect_landscape_orientation, rotate_cv_image
from evergrain.utils import io


class OrientationEngine:
    @staticmethod
    def correct_image_orientation(
        image_path: str | Path,
        save_path: str | Path | None = None,
    ) -> None:
        image_path = Path(image_path)
        angle = detect_landscape_orientation(str(image_path))
        print(f'{image_path}: rotating {angle}°' if angle else f'{image_path}: no rotation')
        if angle == 0:
            return

        img = io.load_image(image_path)
        rotated = img.rotate(-angle, expand=True)
        output_path = Path(save_path) if save_path else image_path
        io.overwrite_image(rotated, output_path)

    @staticmethod
    def visualize_detections(
        image_path: str | Path,
        mode: str = 'face',
        conf_threshold: float = 0.7,
    ) -> None:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            print('Image not found:', image_path)
            return

        rotated_90 = rotate_cv_image(image, 90)

        if mode == 'face':
            boxes_0 = detect_faces(image, conf_threshold)
            boxes_90 = detect_faces(rotated_90, conf_threshold)
        elif mode == 'person':
            boxes_0 = detect_persons(image, conf_threshold)
            boxes_90 = detect_persons(rotated_90, conf_threshold)
        else:
            raise ValueError("Mode must be 'face' or 'person'")

        for x, y, w, h in boxes_0:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for x, y, w, h in boxes_90:
            cv2.rectangle(rotated_90, (x, y), (x + w, y + h), (0, 0, 255), 2)

        base = image_path.stem
        out_dir = image_path.parent
        suffix = 'dnn' if mode == 'face' else 'person_dnn'
        cv2.imwrite(str(out_dir / f'{base}_0_{suffix}.jpg'), image)
        cv2.imwrite(str(out_dir / f'{base}_90_{suffix}.jpg'), rotated_90)
        print(f'{image_path}: DNN {mode}s - 0°: {len(boxes_0)}, 90°: {len(boxes_90)}')
