import cv2
import time
from datetime import datetime
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r"E:\AI\3 July - YOLO\yolov8n.pt")

# Webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Camera not found!")
    exit()

cv2.namedWindow("YOLOv8 Object Detection", cv2.WINDOW_NORMAL)

prev_time = 0

recording = False
writer = None

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Detect ALL objects
        results = model(frame, verbose=False)

        annotated = results[0].plot()

        # -----------------------------
        # Count Persons and Total Objects
        # -----------------------------
        person_count = 0
        total_objects = len(results[0].boxes)

        for box in results[0].boxes:
            cls = int(box.cls[0])

            if cls == 0:      # Person class
                person_count += 1

        # FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # Timestamp
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        cv2.putText(
            annotated,
            f"Persons : {person_count}",
            (20,35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.putText(
            annotated,
            f"Objects : {total_objects}",
            (20,70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,0,0),
            2
        )

        cv2.putText(
            annotated,
            f"FPS : {fps:.2f}",
            (20,105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2
        )

        cv2.putText(
            annotated,
            timestamp,
            (20,140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )

        # Recording
        if recording:

            writer.write(annotated)

            cv2.putText(
                annotated,
                "REC",
                (1100,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

        cv2.imshow("YOLOv8 Object Detection", annotated)

        key = cv2.waitKey(1) & 0xFF

        # ESC or Q
        if key == 27 or key == ord('q'):
            break

        # Screenshot
        elif key == ord('s'):

            filename = datetime.now().strftime("Screenshot_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(filename, annotated)
            print("Saved:", filename)

        # Recording
        elif key == ord('r'):

            if not recording:

                filename = datetime.now().strftime("Recording_%Y%m%d_%H%M%S.mp4")

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                writer = cv2.VideoWriter(
                    filename,
                    fourcc,
                    20,
                    (annotated.shape[1], annotated.shape[0])
                )

                recording = True
                print("Recording Started")

            else:

                recording = False
                writer.release()
                writer = None
                print("Recording Stopped")

finally:

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()