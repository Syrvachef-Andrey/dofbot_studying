import cv2
import numpy as np

color_hsv = {
    'red':    (np.array([0, 43, 46]),    np.array([10, 255, 255])),
    'green':  (np.array([35, 43, 46]),   np.array([77, 255, 255])),
    'blue':   (np.array([100, 43, 46]),  np.array([124, 255, 255])),
    'yellow': (np.array([26, 43, 46]),   np.array([34, 255, 255])),
}

MIN_AREA_RATIO = 0.005          
KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color_mask = cv2.inRange(hsv_img, color_hsv['blue'][0], color_hsv['blue'][1])

    color_mask = cv2.medianBlur(color_mask, 5)

    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN,  KERNEL_OPEN)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)

    contours, _ = cv2.findContours(
        color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        cv2.putText(frame, "no contours", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("frame", frame)
        cv2.imshow("color_mask", color_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    frame_area = frame.shape[0] * frame.shape[1]
    min_area = frame_area * MIN_AREA_RATIO

    if area < min_area:
        cv2.putText(frame, f"too small: {area:.0f} < {min_area:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("frame", frame)
        cv2.imshow("color_mask", color_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    M = cv2.moments(largest_contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cx, cy = x + w // 2, y + h // 2

    cv2.drawContours(frame, [largest_contour], -1, (0, 0, 255), 2)

    x, y, w, h = cv2.boundingRect(largest_contour)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

    side = (w + h) // 2
    cv2.circle(frame, (cx, cy), side // 2, (0, 255, 0), 2)  
    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

    cv2.putText(frame, f"area={area:.0f}  side={side}px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("color_mask", color_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()