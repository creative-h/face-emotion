import numpy as np
import argparse
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plotter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # command line argument
# ap = argparse.ArgumentParser()
# ap.add_argument("--mode",help="train/display")
# a = ap.parse_args()
# mode = a.mode


# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

def displayEmoition():
    # if mode == "display":
    model.load_weights('model.h5')
    # model = load_model('model.h5')
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Not-Interested", 2: "Fearful", 3: "Happy", 4: "Confident", 5: "Sad", 6: "Surprised"}
    net_emotion = {}

    # start the webcam feed
    cap = cv2.VideoCapture('output500ms.mp4')
    print("analysing video......")
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-10), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+10, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            key = emotion_dict[maxindex]
            if key in net_emotion:
                net_emotion[key] += 1
            else:
                net_emotion.update({key:1})
        keyMax = max(net_emotion, key = net_emotion.get)
        all_sum = sum(net_emotion.values())


        # cv2.imshow('Video', cv2.resize(frame,(160,96),interpolation = cv2.INTER_CUBIC))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    print("emotion analysis is :", keyMax)
    normalized_prediction = {k: (v / float(all_sum)*100)  for k,v in net_emotion.items()}

    # cap.release()
    # cv2.destroyAllWindows()
    return normalized_prediction

if __name__== "__main__":
    x = displayEmoition()
    print(x)
    # for key, value in x.items():
    #     print (key, "{:.2f}".format(value) ,"%")
