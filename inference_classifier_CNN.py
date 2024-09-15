import torch
import cv2
import mediapipe as mp
import CNN_model
import numpy as np
import time

input_size = 42
hidden_size = 35
num_classes = 3

model = CNN_model.ConvNet()
model.load_state_dict(torch.load('cnn.pth'))
model.eval()


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m", 13: "n",
               14: "o", 15: "p", 16: "q", 17: "r", 18: "s", 19: "t", 20: "u", 21: "v", 22: "w", 23: "x", 24: "y", 25: "z"}

i = 0
start_time = time.time()
prompt_lines = ['Press "C" to capture!', 'Press "Q" to quit!']

# Font settings
font = cv2.FONT_HERSHEY_COMPLEX
font_scale = 0.75
font_thickness = 3
font_color = (128, 0, 0)  # Green in BGR format
line_height = 30  # Adjust this value to set the vertical spacing between lines

while cap.isOpened():
    tobrk = False
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Initial coordinates
        x, y = 50, 50
        for prompt in prompt_lines:
            cv2.putText(frame, prompt, (x, y), font, font_scale,
                        font_color, font_thickness, cv2.LINE_AA)
            y += line_height
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('c'):
            break
        elif key == ord('q'):
            tobrk = True
            break
    if tobrk:
        break
    elapsed_time = time.time() - start_time
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape

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

        elapsed_time = 100

        if elapsed_time >= 1 and hand_image.shape[0] > 0 and hand_image.shape[1] > 0:
            start_time = time.time()
            # Display the extracted hand image
            # hand_image_gray = hand_image
            hand_image_gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
            hand_image_gray = cv2.resize(hand_image_gray, (200, 200))
            hand_image_gray = hand_image_gray/256
            hand_image_gray = hand_image_gray.astype(np.float32)
            hand_image_gray = np.stack([hand_image_gray] * 3)
            hand_image_gray = hand_image_gray[np.newaxis, :, :, :]
            # hand_image_gray = np.array([hand_image_gray] * 3)
            # hand_image_gray = np.transpose(hand_image_gray, (0, 3, 1, 2))
            # print(hand_image_gray.shape)
            # print(type(hand_image_gray))
            # use model and predict

            outputs = model(torch.tensor(hand_image_gray))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            # print(probs)
            _, predicted = torch.max(probs, 1)

            confidence = probs[0, predicted].item() * 100
            alphabet = labels_dict[predicted.item()]
            print(f"{i}: {alphabet}")
            i += 1

            # if len(prompt_lines) < 2:
            #     word = alphabet
            # else:
            #     word += alphabet
            prompt_lines = [f'Alphabet: {alphabet}',
                            f'Confidence: {confidence: .2f}%',
                            # f'Word till now: {word}',
                            'Press "C" to capture again!',
                            'Press "Q" to quit!',]

    cv2.imshow('frame', frame)


cap.release()
cv2.destroyAllWindows()
