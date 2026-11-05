import cv2
import numpy as np
from Arm_Lib import Arm_Device
import time

cap = cv2.VideoCapture(1)
Arm = Arm_Device()
time.sleep(.1)
time_1 = 500
time_2 = 1000
time_sleep = 0.5

def main():
    cv2.namedWindow("controls")
    cv2.createTrackbar("low",  "controls", 80,  255, lambda x: None)
    cv2.createTrackbar("high", "controls", 180, 255, lambda x: None)
    Arm.Arm_serial_servo_write6(90, 90, 50, -40, 90, 90, 500)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        low  = cv2.getTrackbarPos("low",  "controls")
        high = cv2.getTrackbarPos("high", "controls")
        if low > high:
            low, high = high, low

        mask = cv2.inRange(gray, low, high)

        cv2.imshow("frame", frame)
        cv2.imshow("gray", gray)
        cv2.imshow("mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        print(low, high)
        
    Arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 500) 
    cap.release()
    cv2.destroyAllWindows()

main()

del Arm
    
