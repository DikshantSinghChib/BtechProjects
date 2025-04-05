#Object detection in video survillance system 

import random
import cv2
import numpy as np
from ultralytics import YOLO

# opening the file in read mode
my_file = open("coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")


frame_wid = 640
frame_hyt = 480
line_pos=550
offset=6
count = int(0)

cap = cv2.VideoCapture("video1.mp4")
cap.set(3, frame_wid)  # Set width
cap.set(4, frame_hyt) 
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Cannot able to read frame")
        break

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    cv2.line(frame, (25, line_pos), (1200, line_pos), (255, 127, 0), 3)

    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            cv2.rectangle(
                frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),
                detection_colors[int(clsID)],
                3,
            )

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                (int(bb[0]), int(bb[1]) - 10),
                font,
                1,
                (255, 255, 255),
                2,
            )
            center = center_handle(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
            
            cv2.circle(frame, center, 4, (0, 0, 255))
    # Check for vehicles that crossed the line
    for i in range(len(detect_params[0])):
        boxes = detect_params[0].boxes
        box = boxes[i]
        bb = box.xyxy.numpy()[0]
        x1 = int(bb[0])
        y1 = int(bb[1])
        x2 = int(bb[2])
        y2 = int(bb[3])
    
        # Calculate the center of the bounding box
        center = center_handle(x1, y1, x2 - x1, y2 - y1)
    
        x, y = center
    
        # Check if the center of the vehicle is near the line
        if y > (line_pos - offset) and y < (line_pos + offset):
            count = count + 1
            detection_colors.append((x, y))

    cv2.putText(frame, f"Vehicles Count : {count}", (460, 70),font, 2, (0, 0, 255), 5)
    # Display the resulting frame
    cv2.imshow("ObjectDetection", frame)
    if cv2.waitKey(1) == 13:
        break


cap.release()
cv2.destroyAllWindows()