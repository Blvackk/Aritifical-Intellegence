import cv2

cap = cv2.VideoCapture("highway.mp4")

if not cap.isOpened():
    print("Error: Could not open highway.mp4")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video.")
        break

    cv2.imshow("Frame", frame)

    if cv2.waitKey(30) & 0xFF == 27:   # ESC key
        break

cap.release()
cv2.destroyAllWindows()