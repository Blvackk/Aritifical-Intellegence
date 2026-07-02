import cv2

# Load the video
cap = cv2.VideoCapture("highway.mp4")

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open highway.mp4")
    exit()

# Create background subtractor
object_detector = cv2.createBackgroundSubtractorMOG2(
    history=100,
    varThreshold=40
)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Reached end of video.")
        break

    # Apply background subtraction
    mask = object_detector.apply(frame)

    # Display the original frame and mask
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Exit on pressing ESC
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()