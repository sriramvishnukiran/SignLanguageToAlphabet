import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data_FNN'

count = 0
data = []
labels = []
for dir_ in tqdm(os.listdir(DATA_DIR), desc='Directory', total=len(os.listdir(DATA_DIR)), leave=True):
    for img_path in tqdm(os.listdir(os.path.join(DATA_DIR, dir_)), desc='Image', total=len(os.listdir(os.path.join(DATA_DIR, dir_))), leave=False):
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_rgb,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
                count += 1

            if len(data_aux) > 42:
                tqdm.write(f'{dir_}/{img_path}')
            else:
                data.append(data_aux)
                labels.append(ord(dir_)-97)

plt.show()
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
