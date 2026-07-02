import cv2
from tracker import EuclideanDistTracker

# Create tracker object
tracker = EuclideanDistTracker()

# Load video
cap = cv2.VideoCapture("highway.mp4")

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open highway.mp4")
    exit()

# Background subtractor
object_detector = cv2.createBackgroundSubtractorMOG2(
    history=100,
    varThreshold=40
)

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    # Region of Interest (ROI)
    roi = frame[340:720, 500:800]

    # Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # Object Tracking
    boxes_ids = tracker.update(detections)

    for box_id in boxes_ids:
        x, y, w, h, object_id = box_id

        cv2.rectangle(
            roi,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            roi,
            str(object_id),
            (x, y - 10),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 0),
            2
        )

    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi)
    cv2.imshow("Mask", mask)

    # Exit on ESC
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()