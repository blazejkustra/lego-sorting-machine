import numpy as np
import cv2
import pickle
import pandas as pd
import StackImages

#############################################

brightness = 180
probability_threshold = 0.75
minBlur = 0  # SMALLER VALUE MEANS MORE BLURRINESS PRESENT
minArea = 100
scale = 0.25
scale_saved_images = 0.5
model_path = "model_trained_10epoch_4bricks.p"
font = cv2.FONT_HERSHEY_SIMPLEX

##############################################

cap = cv2.VideoCapture(0)

data = pd.read_csv("labels.csv")
pickle_file = open(model_path, "rb")
model = pickle.load(pickle_file)

##############################################


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


def getClassName(class_number):
    return data["Name"][class_number]


def scaleCoordinate(unscaled):
    return int(unscaled / scale * scale_saved_images)


def recognize(img, img_small, img_contour, img_dilate, img_result):
    contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_SIMPLE

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minArea:
            cv2.drawContours(img_contour, contour, -1, (255, 0, 255), 3)

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)

            if y < int(80*scale) or y+h> int(1000*scale): continue

            cv2.rectangle(img_result, (int(x/scale*scale_saved_images), int(y/scale*scale_saved_images)), (int((x+w)/scale*scale_saved_images), int((y+h)/scale*scale_saved_images)), (0, 0, 255), 2)
            blur = cv2.Laplacian(img_small, cv2.CV_64F).var()

            if blur > minBlur:
                x = scaleCoordinate(x)
                y = scaleCoordinate(y)
                w = scaleCoordinate(w)
                h = scaleCoordinate(h)

                img_boxed = img[y: y + h, x: x + w]

                img_object = np.asarray(img_boxed)
                img_object = cv2.resize(img_object, (32, 32))
                img_object = preprocessing(img_object)
                cv2.imshow("Processed Image", img_object)
                img_object = img_object.reshape(1, 32, 32, 1)

                # PREDICTION
                predictions = model.predict(img_object)
                class_index = model.predict_classes(img_object)
                probability = np.amax(predictions)

                if probability > probability_threshold:
                    cv2.putText(img_result, "CLASS: " + getClassName(int(class_index)), (x, y-35), font, 0.75,(0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(img_result, "PROBABILITY: " + str(round(probability * 100, 2)) + "%", (x, y-10), font, 0.75, (0, 0, 255), 2,cv2.LINE_AA)


def main():
    while True:
        success, img = cap.read()
        if not success:
            print("Can't receive web-cam image")
            break

        img = img[0:1080, 400:1520]

        img_small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        img = cv2.resize(img, (0, 0), fx=scale_saved_images, fy=scale_saved_images)
        img_contour = img_small.copy()
        img_result = img.copy()

        img_blur = cv2.GaussianBlur(img_small, (7, 7), 1)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        img_canny = cv2.Canny(img_gray, 45, 20)

        kernel = np.ones((5, 5))
        img_dilate = cv2.dilate(img_canny, kernel, iterations=1)

        recognize(img, img_small, img_contour, img_dilate, img_result)
        img_stack = StackImages.stackImages(1, ([img_small, img_canny, img_dilate],
                                                [img_contour, img_result, img_result]))

        cv2.imshow("Result", img_stack)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
    cap.release()
    cv2.destroyAllWindows()
