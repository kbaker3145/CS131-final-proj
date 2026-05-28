"""
Scan a video for known faces and build a timeline of when each person
appears and leaves the frame. Results are written to a JSON file.
"""

import os
import face_recognition
import cv2
import numpy as np
import json
from PIL import Image, ImageOps
from datetime import timedelta

# ── config ──────────────────────────────────────────────────────────────────
VIDEO_PATH = "./media/video/main.mov"
KNOWN_PEOPLE_DIR = "./media/known_people"
OUTPUT_JSON = "./runs/face_timeline.json"

FRAME_SKIP = 5          # process every Nth frame (higher = faster but less precise timestamps)
TOLERANCE = 0.75        # lower = stricter matching (default 0.6)
ABSENCE_THRESHOLD = 1.0 # seconds a face must be gone before marking as "left"
MIN_DURATION = 0.7      # appearances shorter than this are excluded from JSON output
# ────────────────────────────────────────────────────────────────────────────


def load_known_faces(directory):
    """Load reference photos and compute a 128-d face encoding for each person."""
    known_encodings, known_names = [], []
    for filename in os.listdir(directory):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        # Derive display name from filename (e.g. "jane_doe.jpg" -> "Jane Doe")
        name = os.path.splitext(filename)[0].replace("-", " ").replace("_", " ").title()
        path = os.path.join(directory, filename)
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # correct orientation from phone/camera EXIF
        img.thumbnail((1000, 1000))         # cap size for faster encoding
        arr = np.array(img)
        encs = face_recognition.face_encodings(arr)
        if not encs:
            print(f"WARNING: No face found for {name}, skipping.")
            continue
        known_encodings.append(encs[0])
        known_names.append(name)
        print(f"✓ Loaded {name}")
    return known_encodings, known_names


def seconds_to_ts(s):
    """Convert seconds to a human-readable timestamp like '0:01:23.45'."""
    return str(timedelta(seconds=round(s, 2)))[:-4]


# ── load reference faces ─────────────────────────────────────────────────────
known_encodings, known_names = load_known_faces(KNOWN_PEOPLE_DIR)

# ── open video ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"\nVideo: {fps:.1f} fps, {total_frames} frames ({seconds_to_ts(total_frames/fps)} duration)")

# Track who is currently on screen and when they were last seen.
# active_faces: { name: {"entered": float, "last_seen": float} }
active_faces = {}
appearances = []  # completed enter/leave records written to JSON at the end

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames we aren't sampling to speed up processing
    if frame_idx % FRAME_SKIP != 0:
        frame_idx += 1
        continue

    timestamp = frame_idx / fps

    # Resize and convert BGR (OpenCV) -> RGB (face_recognition expects RGB)
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # Detect face bounding boxes, then compute encodings for each face
    locations = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, locations)

    seen_this_frame = set()

    for encoding in encodings:
        # Compare this face against all known encodings; pick the closest match
        distances = face_recognition.face_distance(known_encodings, encoding)
        if len(distances) == 0:
            name = "Unknown"
        else:
            best_idx = np.argmin(distances)
            name = known_names[best_idx] if distances[best_idx] <= TOLERANCE else "Unknown"

        seen_this_frame.add(name)

        if name not in active_faces:
            # New face entered the frame
            active_faces[name] = {"entered": timestamp, "last_seen": timestamp}
            print(f"  [{seconds_to_ts(timestamp)}] ENTER: {name}")
        else:
            # Still on screen — update last seen time
            active_faces[name]["last_seen"] = timestamp

    # Faces in active_faces but not seen this frame may have left
    departed = []
    for name, state in active_faces.items():
        if name not in seen_this_frame and (timestamp - state["last_seen"]) >= ABSENCE_THRESHOLD:
            appearances.append({
                "name": name,
                "entered": seconds_to_ts(state["entered"]),
                "left": seconds_to_ts(state["last_seen"]),
                "entered_seconds": round(state["entered"], 2),
                "left_seconds": round(state["last_seen"], 2),
                "duration_seconds": round(state["last_seen"] - state["entered"], 2),
            })
            print(f"  [{seconds_to_ts(timestamp)}]  LEFT: {name} (was on screen for {round(state['last_seen'] - state['entered'], 1)}s)")
            departed.append(name)

    for name in departed:
        del active_faces[name]

    frame_idx += 1
    if frame_idx % (FRAME_SKIP * 30) == 0:
        print(f"  Progress: {frame_idx}/{total_frames} frames ({timestamp:.1f}s)")

cap.release()

# Anyone still on screen when the video ends gets a record through end_time
end_time = total_frames / fps
for name, state in active_faces.items():
    appearances.append({
        "name": name,
        "entered": seconds_to_ts(state["entered"]),
        "left": seconds_to_ts(end_time),
        "entered_seconds": round(state["entered"], 2),
        "left_seconds": round(end_time, 2),
        "duration_seconds": round(end_time - state["entered"], 2),
    })

appearances.sort(key=lambda x: x["entered_seconds"])
appearances = [a for a in appearances if a["duration_seconds"] >= MIN_DURATION]

with open(OUTPUT_JSON, "w") as f:
    json.dump(appearances, f, indent=2)

print(f"\nDone! {len(appearances)} appearance(s) saved to {OUTPUT_JSON}")
