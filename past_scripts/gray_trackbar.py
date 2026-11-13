import cv2
import numpy as np

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise SystemExit("Камера не открылась")

cv2.namedWindow("controls")
cv2.createTrackbar("low",  "controls", 45,  255, lambda x: None)
cv2.createTrackbar("high", "controls", 125, 255, lambda x: None)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    low  = cv2.getTrackbarPos("low",  "controls")
    high = cv2.getTrackbarPos("high", "controls")
    if low > high:
        low, high = high, low

    mask = cv2.inRange(gray, low, high)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            next_, prev_, first_child, parent = hierarchy[i]
            if parent != -1:
                cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 3)

    cv2.imshow("frame", frame)
    cv2.imshow("mask",  mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()