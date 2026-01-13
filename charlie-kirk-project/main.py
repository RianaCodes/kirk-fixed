import cv2 # type: ignore
import cv2
import pygame
import time
import os
from pathlib import Path
from PIL import Image

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image
import pygame # pyright: ignore[reportMissingImports]
import os
from PIL import Image
from pathlib import Path
import time

pygame.mixer.init()
sound = pygame.mixer.Sound(
    "./assets/we-are-charlie-kirk-song.mp3"
)

# -------- MediaPipe FaceLandmarker Setup --------

model_path = "face_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)
spam = Path("assets/spam")
timer = 2.0
timer_started = None
playing = False

cam = cv2.VideoCapture(0)


while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    height, width, depth = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp=image = Image(image_format=Image.ImageFormat.SRGB, data=rgb_frame)
    result = face_landmarker.detect(mp_image)
    face_landmark_points = result.face_landmarks

    if face_landmark_points:
        one_face_landmark_points = face_landmark_points[0]
        
        left = [one_face_landmark_points[145], one_face_landmark_points[159]]
        for landmark_point in left:
            x = int(landmark_point.x * width)
            y = int(landmark_point.y * height)
            print(x, y)
            cv2.circle(frame, (x,y), 3, (0,255,255))


        right = [one_face_landmark_points[374], one_face_landmark_points[386]]
        for landmark_point in right:
                x = int(landmark_point.x * width)
                y = int(landmark_point.y * height)
                print(x, y)
                cv2.circle(frame, (x,y), 3, (255,255,0))

        l_iris = one_face_landmark_points[468]
        r_iris = one_face_landmark_points[473]
        
        l_ratio = (l_iris.y  - left[1].y)  / (left[0].y  - left[1].y  + 1e-6)
        r_ratio = (r_iris.y - right[1].y) / (right[0].y - right[1].y + 1e-6)

        playing = False
        current = time.time()

        if ((l_ratio > 0.70) and (r_ratio > 0.70)):
            if timer_started is None:
                 timer_started = current

            if (current - timer_started) >= timer:
                if not playing:
                    sound.play()
                    for loop in range(4):
                        for i in spam.iterdir():
                            im = Image.open(i)
                            im.show()
                    
                    playing = True

            timer_started = True
                
        else: 
             timer_started = None
            
        if not ((l_ratio > 0.70) and (r_ratio > 0.70)):
             playing = False
             

    cv2.imshow('Face Detection', frame)
    key = cv2.waitKey(100)

    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
