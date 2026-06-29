import numpy as np
import cv2

# Load the cascade for face detection
face_classifier = cv2.CascadeClassifier(
    r"E:\AI\29 June - Haarcascade\Haarcascade\haarcascade_frontalface_default.xml"
)

# Load the image
image = cv2.imread(r"E:\AI\29 June - Haarcascade\image.jpg")

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image.")
    exit()

# Resize image while maintaining aspect ratio
height, width = image.shape[:2]

max_width = 800  # Change this value according to your screen
scale = max_width / width

new_width = int(width * scale)
new_height = int(height * scale)

image = cv2.resize(image, (new_width, new_height))

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_classifier.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=3,
    minSize=(50, 50)
)

print("Faces detected:", len(faces))

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)

# Create a resizable window
cv2.namedWindow("Detected Faces", cv2.WINDOW_NORMAL)

# Display the image
cv2.imshow("Detected Faces", image)

cv2.waitKey(0)
cv2.destroyAllWindows()