import ctypes
import datetime
from model import load_model, predict
import winsound
import cv2
import time

# Reference: https://stackoverflow.com/questions/47377032/python-opencv-detect-eyes-and-save
# Reference 2: https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial

# https://github.com/Itseez/opencv/blob/master/data/haarcascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
open_eyes_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

video_capture = cv2.VideoCapture(1)
left_count = 1
right_count = 1
model = load_model()
is_eye_closed = False
blink_counter = 0
start_time = time.time()
minutes = 1

while True:
    ret, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        open_eyes_glasses = open_eyes_detector.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(open_eyes_glasses) == 2:
            for (ex, ey, ew, eh) in open_eyes_glasses:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                if is_eye_closed:
                    is_eye_closed = False
                    blink_counter += 1
        else:
            left_eyes = left_eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in left_eyes:
                crop_img = roi_color[ey: ey + eh, ex: ex + ew]
                color = (0, 255, 0)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                crop_img = cv2.resize(crop_img, (24, 24))
                pred = predict(crop_img, model)
                if pred == 'closed':
                    color = (0, 0, 255)
                    is_eye_closed = True
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), color, 2)

            right_eyes = right_eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in right_eyes:
                crop_img = roi_color[ey: ey + eh, ex: ex + ew]
                color = (0, 255, 0)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                crop_img = cv2.resize(crop_img, (24, 24))
                pred = predict(crop_img, model)
                if pred == 'closed':
                    color = (0, 0, 255)
                    is_eye_closed = True
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), color, 2)

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    runtime = round(time.time() - start_time, 0)
    if round(time.time() - start_time, 0) > 60:
        minutes = round(time.time() - start_time, 0) / 60
    blinks_per_minute = round(blink_counter / minutes, 1)

    bpm_text_color = (0, 0, 0)
    warned = False
    if blinks_per_minute < 17 and minutes > 1:
        bpm_text_color = (0, 0, 255)
        if runtime % 60 == 1 and warned is False:
            warned = True
            winsound.PlaySound('SystemExclamation', winsound.SND_ASYNC)
            ctypes.windll.user32.MessageBoxW(0, "You need to keep your number of blinks per minute above 17 to avoid "
                                                "eyestrain!", "Warning!", 0)

    if runtime % 60 == 2 and warned:
        warned = False

    txt_size = cv2.getTextSize("Iris has been running for: " + str(datetime.timedelta(seconds=runtime)), font_face,
                               scale, thickness)
    end_x = 30 + txt_size[0][0] + margin
    end_y = 20 - txt_size[0][1] - margin
    cv2.rectangle(img, (10, 40), (end_x, end_y), (255, 255, 255), thickness)
    cv2.putText(img, "Iris has been running for: " + str(datetime.timedelta(seconds=runtime)), (20, 30), font_face,
                scale, (0, 0, 0), 1, cv2.LINE_AA)

    txt_size = cv2.getTextSize("Blinks per minute: " + str(blinks_per_minute) + " (Total blinks = {})"
                               .format(blink_counter), font_face, scale, thickness)
    end_x = 30 + txt_size[0][0] + margin
    end_y = 60 - txt_size[0][1] - margin
    cv2.rectangle(img, (10, 80), (end_x, end_y), (255, 255, 255), thickness)
    cv2.putText(img, "Blinks per minute: " + str(blinks_per_minute) + " (Total blinks = {})"
                .format(blink_counter), (20, 70), font_face, scale, bpm_text_color, 1, cv2.LINE_AA)

    txt_size = cv2.getTextSize("(Optimal number of blinks: > 17)", font_face, scale, thickness)
    end_x = 30 + txt_size[0][0] + margin
    end_y = 100 - txt_size[0][1] - margin
    cv2.rectangle(img, (10, 110), (end_x, end_y), (255, 255, 255), thickness)
    cv2.putText(img, "(Optimal number of blinks per minute: > 17)", (20, 102), font_face, 0.4, bpm_text_color, 1,
                cv2.LINE_AA)

    cv2.imshow('Iris - Eye Strain Prevention (Press ESC to quit)', img)
    k = cv2.waitKey(30) & 0xff
    # Pencet escape utk berhenti
    if k == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
