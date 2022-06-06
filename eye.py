# pip install numpy
# pip install opencv-python

import numpy as np
import cv2

# Reference: https://stackoverflow.com/questions/47377032/python-opencv-detect-eyes-and-save
# Reference 2: https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

video_capture = cv2.VideoCapture(0)
count = 1

while True:
    ret, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    # Note to jap, nek misal kurang, parameter kedua jadikno 1.3

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        left_eyes = left_eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in left_eyes:
            print(count)
            crop_img = roi_color[ey: ey + eh, ex: ex + ew]
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            s1 = 'tmp/left-{}.jpg'.format(count)
            count = count + 1
            cv2.imwrite(s1, crop_img)

        right_eyes = right_eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in right_eyes:
            print(count)
            crop_img = roi_color[ey: ey + eh, ex: ex + ew]
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            s1 = 'tmp/right-{}.jpg'.format(count)
            count = count + 1
            cv2.imwrite(s1, crop_img)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    # Pencet escape utk berhenti
    if k == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
