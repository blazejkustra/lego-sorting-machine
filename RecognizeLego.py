import numpy as np
import cv2
import pickle
import pandas as pd

#############################################

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

minBlur = 0  # SMALLER VALUE MEANS MORE BLURRINESS PRESENT
minArea = 100
scale = 0.5
scaleSaved = 0.75

##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
# IMPORT THE TRANNIED MODEL
data = pd.read_csv("labels.csv")
pickle_in = open("model_trained_10epoch_4bricks.p", "rb")  ## rb = READ BYTE
model = pickle.load(pickle_in)


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getClassName(classNo):
    return data["Name"][classNo]


def recognize():
    contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_SIMPLE for better performance


    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            cv2.drawContours(img_contour, cnt, -1, (255, 0, 255), 3)

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            # if y < int(80*scale) or y+h> int(1000*scale): continue
            cv2.rectangle(img_result, (int(x/scale*scaleSaved), int(y/scale*scaleSaved)), (int((x+w)/scale*scaleSaved), int((y+h)/scale*scaleSaved)), (0, 0, 255), 2)

            blur = cv2.Laplacian(img_small, cv2.CV_64F).var()

            if blur > minBlur:
                x = int(x / scale * scaleSaved)
                y = int(y / scale * scaleSaved)
                w = int(w / scale * scaleSaved)
                h = int(h / scale * scaleSaved)
                img_boxed = img[y: y + h, x: x + w]

                # PROCESS IMAGE
                img_object = np.asarray(img_boxed)
                img_object = cv2.resize(img_object, (32, 32))
                img_object = preprocessing(img_object)
                cv2.imshow("Processed Image", img_object)
                img_object = img_object.reshape(1, 32, 32, 1)
                # PREDICT IMAGE
                predictions = model.predict(img_object)
                classIndex = model.predict_classes(img_object)
                probabilityValue = np.amax(predictions)

                if probabilityValue > threshold:
                    cv2.putText(img_result, "CLASS: " + getClassName(int(classIndex)), (x, y-35), font, 0.75,(0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(img_result, "PROBABILITY: " + str(round(probabilityValue * 100, 2)) + "%", (x, y-10), font, 0.75, (0, 0, 255), 2,cv2.LINE_AA)



while True:
    success, img = cap.read()
    if not success:
        print("Can't receive web-cam image")
        break

    # img = img[0:1080, 400:1520]

    img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    img = cv2.resize(img, (0, 0), fx=scaleSaved, fy=scaleSaved)
    img_contour = img_small.copy()
    img_result = img.copy()

    img_blur = cv2.GaussianBlur(img_small, (7, 7), 1)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 45, 20)

    kernel = np.ones((5, 5))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=1)

    recognize()

    cv2.imshow("Result", img_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
