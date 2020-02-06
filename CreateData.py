import cv2
import numpy as np
import os
import time

#####################################################

myPath = 'data/images'  # Rasbperry Pi:  '/home/pi/Desktop/data/images'
cameraBrightness = 180
moduleVal = 1  # SAVE EVERY ITH FRAME TO AVOID REPETITION
minBlur = 40  # SMALLER VALUE MEANS MORE BLURRINESS PRESENT
minArea = 100
saveData = True  # SAVE DATA FLAG
scale = 0.25
scaleSaved = 0.5

######################################################

cap = cv2.VideoCapture(0)


def empty(a): pass


# CREATE TRACKBAR
cv2.namedWindow("options")
cv2.resizeWindow("options",200,300)
cv2.createTrackbar("first","options",45,250,empty)
cv2.createTrackbar("kernalv","options",5,10,empty)
cv2.createTrackbar("second","options",20,250,empty)

######################################################



def FolderToSave():
    global countFolder
    countFolder = 0
    while os.path.exists(myPath + str(countFolder)):
        countFolder += 1
    os.makedirs(myPath + str(countFolder))


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def saveData(countSave):
    contours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_SIMPLE for better performance

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 3)

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            if y < int(80*scale) or y+h> int(1000*scale): continue
            cv2.rectangle(imgBox, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
            blur = cv2.Laplacian(imgSmall, cv2.CV_64F).var()

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2)
            cv2.putText(imgContour, "Blur: " + str(int(blur)), (x + w + 20, y + 70), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2)

            if saveData:
                if count % moduleVal == 0 and blur > minBlur:
                    nowTime = time.time()
                    x = int(x / scale * scaleSaved)
                    y = int(y / scale * scaleSaved)
                    w = int(w / scale * scaleSaved)
                    h = int(h / scale * scaleSaved)
                    boxedImage = img[y: y + h, x: x + w]
                    cv2.imwrite(myPath + str(countFolder) + '/' + str(countSave) + "_" + str(int(blur)) + "_" + str(
                        nowTime) + ".png", boxedImage)
                    countSave += 1
    return countSave


if saveData: FolderToSave()

count = 0
countSave = 0
while True:
    success, img = cap.read()
    if not success:
        print("Can't receive web-cam image")
        break

    img = img[0:1080,400:1520]

    imgSmall = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    img = cv2.resize(img, (0, 0), fx=scaleSaved, fy=scaleSaved)
    imgContour = imgSmall.copy()
    imgBox = imgSmall.copy()

    imgBlur = cv2.GaussianBlur(imgSmall, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    first = cv2.getTrackbarPos("first", "options")
    second = cv2.getTrackbarPos("second", "options")
    kernalv = cv2.getTrackbarPos("kernalv", "options")

    imgCanny = cv2.Canny(imgGray, first, second)

    kernel = np.ones((kernalv, kernalv))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)

    countSave = saveData(countSave)
    count += 1

    imgStack = stackImages(1, ([imgSmall, imgCanny, imgDil], [imgContour, imgBox, imgBox]))

    print("count:", count, " countSave:", countSave)

    cv2.imshow('result', imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("exit")
        break


cap.release()
cv2.destroyAllWindows()
