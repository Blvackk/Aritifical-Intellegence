import cv2
import numpy as np

# ==========================================
# Load Haar Cascade Classifiers
# ==========================================

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Check if cascades loaded successfully
if face_classifier.empty():
    print("Error: Face Cascade could not be loaded.")
    exit()

if eye_classifier.empty():
    print("Error: Eye Cascade could not be loaded.")
    exit()

# ==========================================
# Load Image
# ==========================================

image = cv2.imread(r"E:\AI\29 June - Haarcascade\image2.jpg")

if image is None:
    print("Error: Could not load image.")
    exit()

# ==========================================
# Resize Image
# ==========================================

height, width = image.shape[:2]

max_width = 800

scale = max_width / width

new_width = int(width * scale)
new_height = int(height * scale)

image = cv2.resize(image, (new_width, new_height))

# ==========================================
# Convert to Grayscale
# ==========================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Improve contrast
gray = cv2.equalizeHist(gray)

# ==========================================
# Detect Faces
# ==========================================

faces = face_classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)

print("Faces Detected :", len(faces))

# ==========================================
# Detect Eyes
# ==========================================

for (x, y, w, h) in faces:

    # Draw face rectangle
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 2)

    # Face ROI
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]

    # Equalize face ROI
    roi_gray = cv2.equalizeHist(roi_gray)

    # Detect Eyes
    eyes = eye_classifier.detectMultiScale(
        roi_gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(15, 15)
    )

    print("Eyes Detected :", len(eyes))

    # Draw eye rectangles
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(
            roi_color,
            (ex, ey),
            (ex+ew, ey+eh),
            (0, 255, 0),
            2
        )

# ==========================================
# Show Image
# ==========================================

cv2.namedWindow("Face and Eye Detection", cv2.WINDOW_NORMAL)

cv2.imshow("Face and Eye Detection", image)

cv2.waitKey(0)

cv2.destroyAllWindows()