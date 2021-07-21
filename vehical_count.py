import cv2
import numpy as np

capture = cv2.VideoCapture("D:\\Pragatheeshwar\\pycharm_project\\vehical video.mp4")

min_width_rect = 80  # minimum width of rectangle
min_height_rect = 80
count_line_position = 550

ag = cv2.createBackgroundSubtractorMOG2()


def centre_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


detect = []
offset = 6
counter = 0
while True:
    success, cap = capture.read()
    cv2.putText(cap,"BY: Pragatheeshwar",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
    grey = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    # apply on all vehicals
    image = ag.apply(blur)  # applied to all, blurred all
    dilat = cv2.dilate(image, np.ones((5, 5)))
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilat___ = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernal)
    dilat___ = cv2.morphologyEx(dilat___, cv2.MORPH_CLOSE, kernal)
    contors, h = cv2.findContours(dilat___, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(cap, (25, count_line_position), (1200, count_line_position), (254, 92, 57), 2)

    for (i, c) in enumerate(contors):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rect) and (h > min_height_rect)
        if not validate_counter:
            continue
        cv2.rectangle(cap, (x, y), (x + w, y + h), (254, 92, 57), 2)
        cv2.putText(cap, "VEHICAL ID: " + str(counter), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        centre = centre_handle(x, y, w, h)
        detect.append(centre)
        cv2.circle(cap, centre, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                counter += 1
            #cv2.line(cap, (25, count_line_position), (1200, count_line_position), (254, 92, 57), 2)
            detect.remove((x, y))
            print("vehical counter: " + str(counter))

    cv2.putText(cap, "NUMBER OF VEHICALS: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow('detector', dilat___)
    cv2.imshow('Vehical video', cap)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
