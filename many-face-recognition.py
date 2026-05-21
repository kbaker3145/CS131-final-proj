import face_recognition
from PIL import Image, ImageOps
import numpy as np
import os

# likely wont need this if we screenshot from a video
def load_and_resize(path, max_size=1000):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img.thumbnail((max_size, max_size))
    return np.array(img)

# known face encodings
known_faces = {
    "Kate Baker": load_and_resize("./media/known_people/kate-baker.JPG"),
    "Yasmine Alonso": load_and_resize("./media/known_people/yasmine-alonso.JPG"),
}

known_encodings = []
known_names = []

for name, image in known_faces.items():
    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        print(f"WARNING: No face found in reference photo for {name}. Skipping.")
        continue
    known_encodings.append(encodings[0])
    known_names.append(name)
    print(f"✓ Loaded reference face for {name}")

# loop through unknown photos 
unknown_dir = "./media/unknown_pictures"

for filename in os.listdir(unknown_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(unknown_dir, filename)
    image = load_and_resize(path)

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    print(f"\n{filename}: found {len(face_locations)} face(s)")

    for i, (face_encoding, location) in enumerate(zip(face_encodings, face_locations)):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        distances = face_recognition.face_distance(known_encodings, face_encoding)

        if True in matches:
            best_match_index = np.argmin(distances)
            name = known_names[best_match_index]
            confidence = 1 - distances[best_match_index]
            print(f"  Face {i+1}: {name} (confidence: {confidence:.1%})")
        else:
            print(f"  Face {i+1}: Unknown person")