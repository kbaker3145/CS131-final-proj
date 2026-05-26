# CS131-final-proj
By Yasmine Alonso and Kate Baker

## Info

Project guidelines: https://docs.google.com/document/d/1TCWvwTsr0wCwhIP1v8HKqBlrvbQLdqNCM5sl24weqgA/edit?tab=t.0#heading=h.st7n01sbf1iz

Our project proposal: https://docs.google.com/document/d/1ZXEhaz3OlIZP2Tak4T0a8P6yQ8CuhQNwRiHmeOGpZdU/edit?tab=t.0


## Repo setup 
source .venv/bin/activate

### Object recognition (sink dish tracking)

https://github.com/ultralytics/ultralytics (YOLOv8)

Track add/remove events for dishes in a user-defined sink ROI:

```bash
pip install -r requirements.txt

# First time: draw ROI on first frame to clip video processing to just the sink (saved for reuse)
python track_sink_dishes.py \
  --video /path/to/sink_clip.MOV \
  --pick-roi \
  --save-roi config/sink_roi.json \
  -o output/sink_events.json

# Later runs
python track_sink_dishes.py \
  --video /path/to/sink_clip.MOV \
  --roi-file config/sink_roi.json \
  -o output/sink_events.json
```

Optional: `--pick-roi-from-image Screenshot.png` if the camera angle matches the video.

For longer videos with high frame rates, you can use the following flags to make video processing faster (e.g. `--stride 3` or `--stride 5`) to speed up processing!

Sample output should loook something like this:
```
{
  "video": "/Users/yasminealonso/Downloads/IMG_3107.MOV",
  "fps": 59.998,
  "frame_count_reported": 2491,
  "frames_processed": 100,
  "frame_stride": 5,
  "roi": {
    "x": 518,
    "y": 844,
    "w": 1123,
    "h": 1843
  },
  "model": "yolov8s.pt",
  "conf": 0.35,
  "kitchen_classes": [
    "bottle",
    "bowl",
    "cup",
    "fork",
    "knife",
    "spoon",
    "wine glass"
  ],
  "parameters": {
    "min_present_frames": 8,
    "min_absent_frames": 15,
    "baseline_sec": 2.0
  },
  "events": [
    {
      "type": "added",
      "track_id": 24,
      "class": "spoon",
      "time_sec": 2.917,
      "frame": 175
    },
    {
      "type": "removed",
      "track_id": 24,
      "class": "spoon",
      "time_sec": 6.0,
      "frame": 360
    }
  ],
  "tracks": {
    "24": {
      "class": "spoon",
      "status": "removed",
      "first_seen_sec": 2.333,
      "last_seen_sec": 4.75,
      "first_seen_frame": 140,
      "last_seen_frame": 285,
      "added_sec": 2.917,
      "added_frame": 175,
      "removed_sec": 6.0,
      "removed_frame": 360
    }
  },
  "summary": {
    "add_count": 1,
    "remove_count": 1,
    "fixtures_ignored": 0
  }
}
```


### Face recognition

https://github.com/ageitgey/face_recognition

- MIT licensed, fully open source
- 56k GitHub stars and 13.7k forks aka well-vetted
- The underlying model (dlib's ResNet) runs entirely locally
- Last release was 2018, so it's not actively maintained but it's the underlying dlib is still maintained
- One known issue: accuracy may vary between ethnic groups, and it doesn't work very well on children


first run:


IMG_7156.JPG: found 1 face(s)
  Face 1: Kate Baker (confidence: 41.7%)

IMG_2167.JPG: found 2 face(s)
  Face 1: Kate Baker (confidence: 51.1%)
  Face 2: Yasmine Alonso (confidence: 61.1%)

IMG_2559.JPG: found 2 face(s)
  Face 1: Yasmine Alonso (confidence: 53.7%)
  Face 2: Kate Baker (confidence: 51.7%)

IMG_1468.JPG: found 2 face(s)
  Face 1: Yasmine Alonso (confidence: 53.2%)
  Face 2: Kate Baker (confidence: 46.4%)

IMG_6968.JPG: found 2 face(s)
  Face 1: Unknown person
  Face 2: Kate Baker (confidence: 46.7%)

IMG_3166.JPG: found 2 face(s)
  Face 1: Yasmine Alonso (confidence: 58.8%)
  Face 2: Kate Baker (confidence: 46.1%)

IMG_5762.JPG: found 1 face(s)
  Face 1: Yasmine Alonso (confidence: 63.2%)

IMG_2582.JPG: found 2 face(s)
  Face 1: Unknown person
  Face 2: Yasmine Alonso (confidence: 43.3%)

IMG_5748.JPG: found 1 face(s)
  Face 1: Kate Baker (confidence: 53.0%)

IMG_2387.JPG: found 1 face(s)
  Face 1: Kate Baker (confidence: 54.0%)

IMG_5758.JPG: found 1 face(s)
  Face 1: Yasmine Alonso (confidence: 47.7%)

IMG_5997.JPG: found 1 face(s)
  Face 1: Kate Baker (confidence: 48.3%)