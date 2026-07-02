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

    # Green color range
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Create mask
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Extract green color
    green = cv2.bitwise_and(frame, frame, mask=green_mask)

    # Display results
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Green Mask", green_mask)
    cv2.imshow("Green Color Detection", green)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()