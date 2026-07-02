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

    # Blue color range
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # Create mask
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Extract blue color
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Display results
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Blue Mask", blue_mask)
    cv2.imshow("Blue Color Detection", blue)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()