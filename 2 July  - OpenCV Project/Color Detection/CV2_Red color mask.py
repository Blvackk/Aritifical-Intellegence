import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Lower red range
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    # Upper red range
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    red_mask = mask1 + mask2

    # Extract red color
    red = cv2.bitwise_and(frame, frame, mask=red_mask)

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Red Color Detection", red)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()