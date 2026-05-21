import face_recognition
from PIL import Image
import numpy as np

picture_of_me = face_recognition.load_image_file("./media/known_people/kate-baker.JPG")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# may struggle with larger images? 
def load_and_resize(path, max_size=1000):
    img = Image.open(path)
    # Fix EXIF rotation (common with phone photos)
    from PIL import ImageOps
    img = ImageOps.exif_transpose(img)
    # Resize if too large
    img.thumbnail((max_size, max_size))
    return np.array(img)


# this example: https://github.com/ageitgey/face_recognition/blob/master/examples/recognize_faces_in_pictures.py
picture_of_me = load_and_resize("./media/known_people/kate-baker.JPG")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

unknown_picture = load_and_resize("./media/unknown_pictures/IMG_7156.JPG")
encodings = face_recognition.face_encodings(unknown_picture)

if len(encodings) == 0:
    print("No face found in the unknown picture. Try a clearer/closer shot.")
else:
    unknown_face_encoding = encodings[0]
    results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
    if results[0]:
        print("It's a picture of me!")
    else:
        print("It's not a picture of me!")