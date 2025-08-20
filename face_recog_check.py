import dlib
import numpy as np
from PIL import Image

img_path = r"D:\FSAPP\sample_out\img_1.jpg"
img = np.array(Image.open(img_path).convert("RGB"))

detector = dlib.get_frontal_face_detector()
faces = detector(img, 1)
print("Faces detected:", len(faces))
