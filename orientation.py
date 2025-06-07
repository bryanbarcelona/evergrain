import cv2
from PIL import Image
import os
import numpy as np

# Construct paths to model files
model_dir = os.path.join(os.path.dirname(__file__), "models")
prototxt_path = os.path.join(model_dir, "deploy.prototxt")
caffemodel_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

# Load the face detection model
face_net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

def count_faces_dnn(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    count = 0
    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > 0.7:
            count += 1
    return count

def count_faces_haar(image, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces)

def detect_best_rotation(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return 0

    height, width = image.shape[:2]
    if width > height:
        # Already landscape — skip rotation
        print(f"{image_path}: Already landscape, skipping rotation")
        return 0
    
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # Try DNN first
    dnn_0 = count_faces_dnn(image)
    dnn_90 = count_faces_dnn(rotated)

    if dnn_0 > 0 or dnn_90 > 0:
        return 90 if dnn_90 > dnn_0 else 0

    # Fall back to Haar only if DNN found nothing
    print(f"No DNN detections, using Haar: {image_path}")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    haar_0 = count_faces_haar(image, face_cascade)
    haar_90 = count_faces_haar(rotated, face_cascade)

    return 90 if haar_90 > haar_0 else 0

def rotate_cv_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image  # Only 90° supported

def detect_landscape_orientation(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    if width > height:
        # Already landscape — skip rotation
        print(f"{image_path}: Already landscape, skipping rotation")
        return 0
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def count_faces(rotated_image):
        faces = face_cascade.detectMultiScale(rotated_image, scaleFactor=1.1, minNeighbors=5)
        return len(faces)

    faces_0 = count_faces(gray)
    faces_90 = count_faces(rotate_cv_image(gray, 90))
    print(f"{image_path}: faces_0={faces_0}, faces_90={faces_90}")
    if faces_90 > faces_0 and faces_90 > 0:
        return 90  # it's a sideways landscape
    return 0  # it's already portrait

def auto_correct_photo_orientation(image_path, save_path=None):
    angle = detect_best_rotation(image_path)
    print(f"{image_path}: rotating {angle}°" if angle else f"{image_path}: no rotation")

    if angle == 0:
        return

    img = Image.open(image_path)
    rotated = img.rotate(-angle, expand=True)

    if save_path:
        rotated.save(save_path)
    else:
        rotated.save(image_path)

def draw_detected_faces(image_path, output_folder=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    gray_90 = cv2.cvtColor(rotated_90, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def draw_faces(img, gray_img):
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img, len(faces)

    drawn_0, faces_0 = draw_faces(image.copy(), gray)
    drawn_90, faces_90 = draw_faces(rotated_90.copy(), gray_90)

    # Save to same directory or optional folder
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = output_folder or os.path.dirname(image_path)
    cv2.imwrite(os.path.join(out_dir, f"{base}_0_debug.jpg"), drawn_0)
    cv2.imwrite(os.path.join(out_dir, f"{base}_90_debug.jpg"), drawn_90)

    print(f"{image_path}: Faces at 0° = {faces_0}, at 90° = {faces_90}")

def detect_faces_dnn(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Adjust threshold as needed
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes

def draw_faces_debug_dnn(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found:", image_path)
        return
    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    faces_0 = detect_faces_dnn(image)
    faces_90 = detect_faces_dnn(rotated_90)

    for (x, y, w, h) in faces_0:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in faces_90:
        cv2.rectangle(rotated_90, (x, y), (x + w, y + h), (0, 0, 255), 2)

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.dirname(image_path)
    cv2.imwrite(os.path.join(out_dir, f"{base}_0_dnn.jpg"), image)
    cv2.imwrite(os.path.join(out_dir, f"{base}_90_dnn.jpg"), rotated_90)

    print(f"{image_path}: DNN faces - 0°: {len(faces_0)}, 90°: {len(faces_90)}")