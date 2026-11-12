import cv2
import numpy as np
from Arm_Lib import Arm_Device
import time


Arm = Arm_Device()
time.sleep(.1)

Arm.Arm_serial_servo_write6(90, 90, 50, -40, 90, 90, 5000)

time.sleep(2)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("Камера отвалилась, ждём...")
        time.sleep(0.5)
        continue   # или break, если хочешь выйти
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = cv2.inRange(gray, 16, 95)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(
mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None or len(contours) == 0:
        continue
    hierarchy = hierarchy[0]

    outer_idxs = [i for i in range(len(contours)) if hierarchy[i][3] == -1]
    if not outer_idxs:
        continue
    outer_idx = max(outer_idxs, key=lambda i: cv2.contourArea(contours[i]))
    outer = contours[outer_idx]
    outer_area = cv2.contourArea(outer)

    if outer_area < 500:
        continue

    inner_idxs = [
        i for i in range(len(contours))
        if hierarchy[i][3] == outer_idx
    ]
    if not inner_idxs:
        continue

    inner_idx = max(inner_idxs, key=lambda i: cv2.contourArea(contours[i]))
    inner = contours[inner_idx]
    inner_area = cv2.contourArea(inner)

    ratio = inner_area / outer_area
    if ratio < 0.1:  
        continue

    M = cv2.moments(inner)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)
    
    cv2.drawContours(frame, contours, inner_idx, (0, 255, ), 3)
    
    cv2.imshow("mask", mask)
    cv2.imshow("result", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 500)
cv2.destroyAllWindows()
cap.release()

del Arm