import os
import mediapipe as mp
import cv2
import numpy as np
from tqdm import tqdm

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

MANUAL_CAPTURE = True

DATA_DIR = './data_FNN'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classes_list = [chr(i) for i in range(97, 97+26)]
dataset_size = 50

# Font settings
font = cv2.FONT_HERSHEY_COMPLEX
font_scale = 0.75
font_thickness = 3
font_color = (128, 0, 0)  # Green in BGR format
line_height = 30  # Adjust this value to set the vertical spacing between lines

# Prompt
x, y = 100, 50

cap = cv2.VideoCapture(0)
for j in classes_list:
    prompt = 'Ready? Press "S" ! :)'
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, prompt, (x, y), font, font_scale,
                    font_color, font_thickness, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('s'):
            break

    for counter in tqdm(range(dataset_size), desc="Dataset created", total=dataset_size):
        while MANUAL_CAPTURE:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            prompt = 'Press "C" to capture!'
            cv2.putText(frame, prompt, (x, y), font, font_scale,
                        font_color, font_thickness, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('c'):
                break
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Get bounding box coordinates
                x_min, y_min = int(min(landmark.x for landmark in landmarks.landmark) * frame.shape[1]), \
                    int(min(landmark.y for landmark in landmarks.landmark)
                        * frame.shape[0])
                x_max, y_max = int(max(landmark.x for landmark in landmarks.landmark) * frame.shape[1]), \
                    int(max(landmark.y for landmark in landmarks.landmark)
                        * frame.shape[0])
                y_min = int(y_min*0.8)
                y_max = int(y_max*1.2)
                x_min = int(x_min*0.8)
                x_max = int(x_max*1.2)
                hand_image = frame[y_min:y_max, x_min:x_max]
            if hand_image.shape[0] > 0 and hand_image.shape[1] > 0:
                # hand_image_gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
                hand_image_gray = hand_image
                hand_image_gray = cv2.resize(hand_image_gray, (200, 200))
                cv2.waitKey(100)

                edge_detection_kernel = np.array([[-1, -1, -1],
                                                  [-1, 8, -1],
                                                  [-1, -1, -1]])

                edges = cv2.filter2D(hand_image_gray, -1,
                                     edge_detection_kernel)

                sharpen_detection_kernel = np.array([[0, -1, 0],
                                                     [-1, 5, -1],
                                                     [0, -1, 0]])

                shapened = cv2.filter2D(
                    hand_image_gray, -1, sharpen_detection_kernel)

                cv2.imwrite(os.path.join(DATA_DIR, str(
                    j), f'{counter}.jpg'), hand_image_gray)
                cv2.imwrite(os.path.join(DATA_DIR, str(
                    j), f'{counter}_sharp.jpg'), shapened)


cap.release()
cv2.destroyAllWindows()
