from ultralytics import YOLO

# Load pretrained model — auto-downloads weights on first run
# Sizes: yolov8n (fastest) → yolov8s → yolov8m → yolov8l → yolov8x (most accurate)
model = YOLO("yolov8s.pt")

img = "./media/objects/bowl.png"
results = model(img, conf=0.5, iou=0.45)


# COCO doesn't have "plate" — use these proxies for kitchen items
kitchen_classes = {"bowl", "cup", "fork", "knife", "spoon", "bottle"}

for r in results:
    r.show()

    # Use pandas to avoid polars boolean mask issue
    df = r.to_df()  # polars df
    dishes = df.filter(df["name"].is_in(kitchen_classes))

    print(f"\nAll detections:")
    print(df.select(["name", "confidence"]))

    print(f"\nKitchen items only ({len(dishes)} found):")
    print(dishes.select(["name", "confidence"]))

    if len(dishes) == 0:
        print("No kitchen items detected — note: 'plate' is not a COCO class, try with bowls/cups")