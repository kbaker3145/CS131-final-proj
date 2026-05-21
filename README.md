# CS131-final-proj

## Info

Project guidelines: https://docs.google.com/document/d/1TCWvwTsr0wCwhIP1v8HKqBlrvbQLdqNCM5sl24weqgA/edit?tab=t.0#heading=h.st7n01sbf1iz

YK project proposal: https://docs.google.com/document/d/1ZXEhaz3OlIZP2Tak4T0a8P6yQ8CuhQNwRiHmeOGpZdU/edit?tab=t.0


## Repo setup 

source .venv/bin/activate


### Object recognition

https://github.com/ultralytics/yolov5

this still needs work LOL. we might need to train on our own pics of objects. 



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