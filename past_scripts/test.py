import cv2
from Arm_Lib import Arm_Device
import time
import numpy as np


Arm = Arm_Device()
time.sleep(.1)

try:

    Arm.Arm_serial_servo_write6(90, 90, 40, -40, 90, 90, 1500)

    time.sleep(4)

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not ret:
            break

        mask = cv2.inRange(gray, 45, 125)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierachy is not None:

            hierachy = hierachy[0]

            for i, cnt in enumerate(contours):
                next_, prev_, first_child, parent = hierachy[i]

                if parent != -1:
                    cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 3)

        # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        
        cv2.imshow("frame", frame)
        cv2.imshow("grey_frame", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    Arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1500)

    del Arm