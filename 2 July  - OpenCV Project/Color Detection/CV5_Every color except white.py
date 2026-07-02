import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Convert BGR image to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White color range
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])

    # Create white mask
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Invert the mask (everything except white)
    non_white_mask = cv2.bitwise_not(white_mask)

    # Extract all colors except white
    result = cv2.bitwise_and(frame, frame, mask=non_white_mask)

    # Display results
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Non-White Mask", non_white_mask)
    cv2.imshow("All Colors Except White", result)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()