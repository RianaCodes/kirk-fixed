import cv2
import pygame
import time
from pathlib import Path
from PIL import Image

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage
from mediapipe import ImageFormat

# ---------- Pygame Setup ----------
pygame.mixer.init()
sound = pygame.mixer.Sound(
    "./assets/we-are-charlie-kirk-song.mp3"
)

# ---------- MediaPipe FaceLandmarker Setup ----------
model_path = "face_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)

# ---------- App State ----------
spam = Path("assets/spam")
timer = 2.0
timer_started = None
playing = False

cam = cv2.VideoCapture(0)

# ---------- Main Loop ----------
while True:
    ret, frame = cam.read()
    sound.play()
    time.sleep(1)
    if not ret:
        print("Failed to grab frame from camera")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = MPImage(
        image_format=ImageFormat.SRGB,
        data=rgb_frame
    )

    result = face_landmarker.detect(mp_image)
    face_landmark_points = result.face_landmarks

    if face_landmark_points:
        one_face_landmark_points = face_landmark_points[0]

        left = [
            one_face_landmark_points[145],
            one_face_landmark_points[159]
        ]

        right = [
            one_face_landmark_points[374],
            one_face_landmark_points[386]
        ]

        for landmark_point in left + right:
            x = int(landmark_point.x * width)
            y = int(landmark_point.y * height)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        l_iris = one_face_landmark_points[468]
        r_iris = one_face_landmark_points[473]

        l_ratio = (l_iris.y - left[1].y) / (left[0].y - left[1].y + 1e-6)
        r_ratio = (r_iris.y - right[1].y) / (right[0].y - right[1].y + 1e-6)
        print(f"L_ratio={l_ratio:.3f}, R_ratio={r_ratio:.3f}")

        current = time.time()

        if l_ratio < 0.25 and r_ratio < 0.25:
            if timer_started is None:
                timer_started = current

            if (current - timer_started) >= timer and not playing:
                sound.play()
                for _ in range(4):
                    for img in spam.iterdir():
                        Image.open(img).show()
                playing = True
        else:
            timer_started = None
            playing = False

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
