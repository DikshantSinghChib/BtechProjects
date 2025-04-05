import os                               # provides and operating system 
import cv2                              #used for image processing 
import matplotlib.pyplot as plt        
import mediapipe as mp                  #used for hand ditection 
import pickle                           #for file operation


mp_hands = mp.solutions.hands                                       #use for landmarks of hands on ech image
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'            #all directory from data folder    

data =[]                        #data and lables two empty list 
                                #data used for storing coordinates
labels =[]                      #label used for label the directory
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux =[]

        X =[]
        Y =[]

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))                #convert the image from bgr to rgb
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)                        #result store the hands dimension in the perticular image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x =hand_landmarks.landmark[i].x
                    y =hand_landmarks.landmark[i].y
                    X.append(x)
                    Y.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x =hand_landmarks.landmark[i].x
                    y =hand_landmarks.landmark[i].y
                    data_aux.append(x -min(X))
                    data_aux.append(y -min(Y))

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')                       #create a file data.pricles which contains data and lables

pickle.dump({'data': data, 'labels': labels}, f)

f.close()