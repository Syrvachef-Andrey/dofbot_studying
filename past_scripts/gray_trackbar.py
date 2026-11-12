import cv2
import numpy as np

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise SystemExit("Камера не открылась")

cv2.namedWindow("controls")
cv2.createTrackbar("param2",   "controls", 22, 60, lambda x: None)
cv2.createTrackbar("minR",     "controls", 12, 80, lambda x: None)
cv2.createTrackbar("maxR",     "controls", 60, 150, lambda x: None)
cv2.createTrackbar("fill_thr", "controls", 40, 100, lambda x: None)  # в %

def circle_fill(mask, x, y, r, inner=0.8):
    roi = np.zeros_like(mask)
    cv2.circle(roi, (x, y), max(1, int(r * inner)), 255, -1)
    total = cv2.countNonZero(roi)
    if total == 0:
        return 0.0
    return cv2.countNonZero(cv2.bitwise_and(mask, roi)) / total

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg   = cv2.GaussianBlur(gray, (0, 0), 30)
    norm = cv2.divide(gray, bg, scale=255)
    norm_blur = cv2.medianBlur(norm, 5)

    # маска для верификации
    th = cv2.adaptiveThreshold(norm, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 50)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  k)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)

    # param2 = cv2.getTrackbarPos("param2",   "controls")
    # minR   = max(3, cv2.getTrackbarPos("minR", "controls"))
    # maxR   = max(minR + 1, cv2.getTrackbarPos("maxR", "controls"))
    # fill_t = cv2.getTrackbarPos("fill_thr", "controls") / 100.0
    param2 = 21
    minR   = 7
    maxR   = 24
    fill_t = 0.2

    print(param2, minR, maxR, fill_t)

    circles = cv2.HoughCircles(
        norm_blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=max(20, minR * 2),
        param1=100, param2=max(1, param2),
        minRadius=minR, maxRadius=maxR
    )

    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            if circle_fill(th, x, y, r) >= fill_t:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("frame", frame)
    cv2.imshow("mask",  th)
    cv2.imshow("norm",  norm)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()