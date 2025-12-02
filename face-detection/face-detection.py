# https://github.com/serengil/deepface
from deepface import DeepFace
import os, shutil

detector = "retinaface"
align = True
enforce_detection = False

all_images = "./all_images"
face_images = "./face_images"
os.makedirs(face_images, exist_ok = True)

for file in os.listdir(all_images):
	path = os.path.join(all_images, file)
	face_objs = DeepFace.extract_faces(
		img_path = path, detector_backend = detector, align = align, enforce_detection = enforce_detection
	)
	if len(face_objs) > 0 and max(i['confidence'] for i in face_objs) > 0.5:
		shutil.copy(path, os.path.join(face_images, file))
print("Face images identified.")
