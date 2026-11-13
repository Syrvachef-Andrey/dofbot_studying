import cv2
import numpy as np

color_hsv = {
    'red':    (np.array([0, 43, 46]),    np.array([10, 255, 255])),
    'green':  (np.array([35, 43, 46]),   np.array([77, 255, 255])),
    'blue':   (np.array([100, 43, 46]),  np.array([124, 255, 255])),
    'yellow': (np.array([26, 43, 46]),   np.array([34, 255, 255])),
}

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color_mask = cv2.inRange(hsv_img, color_hsv['blue'][0], color_hsv['blue'][1])

    contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

    M = cv2.moments(largest_contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0

    cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)

    x, y, w, h = cv2.boundingRect(largest_contour)

    cv2.circle(frame, (cx, cy), h // 2, (0, 255, 0), 3)

    cv2.imshow("frame", frame)
    cv2.imshow("color_mask", color_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()