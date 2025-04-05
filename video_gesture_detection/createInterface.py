import numpy as np
import pickle
import mediapipe as mp
import cv2

model_dict = pickle.load(open('./model.p', 'rb'))               # Loading the model
model = model_dict['model']

cap = cv2.VideoCapture(0)       # Opening camera

image_path = r'C:\Users\DIKSHANT\Desktop\video_gesture_detection\image.png'         # Image path for demonstrating the chart
image = cv2.imread(image_path)

if image is None:                           # Edge case for checking if image is opened
    print("Error loading image")
else:
    cv2.imshow('Static Image', image)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

gesture_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N',
                  14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from webcam")                #if the frame not capture halt the program
        break  

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,                      # image to draw
                hand_landmarks,             # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        landmark_data = []
        x_coords = []
        y_coords = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_coords.append(x)
                y_coords.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                landmark_data.append(x - min(x_coords))
                landmark_data.append(y - min(y_coords))

        x1 = int(min(x_coords) * W) - 10
        y1 = int(min(y_coords) * H) - 10

        x2 = int(max(x_coords) * W) - 10
        y2 = int(max(y_coords) * H) - 10

        prediction = model.predict([np.asarray(landmark_data)])

        recognized_gesture = gesture_labels[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (139, 0, 0), 4)  
        cv2.putText(frame, recognized_gesture, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (139, 0, 0), 3, cv2.LINE_AA)  

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):                # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
