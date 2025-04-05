import os
import cv2

data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

#intializing number of classes for collecting data
numclasses = 4
dataset_size = 100

cap = cv2.VideoCapture(0)  # Use camera index 0, or adjust as needed
for j in range(numclasses):
    if not os.path.exists(os.path.join(data_dir, str(j))):
        os.makedirs(os.path.join(data_dir, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while not done:
        ret, frame = cap.read()
        cv2.putText(frame, 'if ready Press "r"', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,cv2.LINE_AA)
        cv2.imshow('frame1', frame)  # Use different window name for clarity
        if cv2.waitKey(25) == ord('r'):
            done = True  

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame2', frame)  # Use different window name for clarity
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_dir, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1   #increment the count 

cap.release()
cv2.destroyAllWindows()
