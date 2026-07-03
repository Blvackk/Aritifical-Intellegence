import ultralytics
ultralytics.checks()

from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt")

# Perform detection and save the output image
results = model.predict(
    source=r"E:\AI\3 - YOLO\Object Detection Using YOLO in CPU\image.jpg",
    save=True,
    show=True
)

print("Saved in:", results[0].save_dir)