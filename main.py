import cv2
from Arm_Lib import Arm_Device
import time
import numpy as np
from itertools import combinations


def draw_line_beetween_two_centers(point1, point2):
    cv2.line(frame, point1, point2, (0, 255, 0), 1)
    len_of_vector = (abs(point1[0] - point2[0]) ** 2 + abs(point1[1] - point2[1]) ** 2) ** 0.5
    print(f"Difference in ppx = {len_of_vector}")

def drawing_middle_of_square(points):
    x_center, y_center = sum([x[0] for x in points]) / 4, sum([y[1] for y in points]) / 4
    x_center, y_center = int(round(x_center)), int(round(y_center))
    cv2.circle(frame, (x_center, y_center), 3, (255, 0, 0), 3)
    return (x_center, y_center)

def circle_fill(mask, x, y, r, inner=0.9):
    roi = np.zeros_like(mask)
    cv2.circle(roi, (x, y), max(1, int(r * inner)), 255, -1)
    total = cv2.countNonZero(roi)
    if total == 0:
        return 0.0
    return cv2.countNonZero(cv2.bitwise_and(mask, roi)) / total

def square_score(pts):
    d = []
    for i in range(4):
        for j in range(i + 1, 4):
            d.append(float(np.hypot(pts[i][0] - pts[j][0],
                                    pts[i][1] - pts[j][1])))
    d = np.sort(d)
    sides, diags = d[:4], d[4:]

    s_mean, d_mean = sides.mean(), diags.mean()
    if s_mean < 1e-6:
        return 1e9

    var_sides  = sides.std() / s_mean
    var_diags  = diags.std() / d_mean
    ratio_err  = abs(d_mean / s_mean - np.sqrt(2.0)) / np.sqrt(2.0)
    return var_sides + var_diags + ratio_err

def find_star(points, score_threshold=0.35):
    n = len(points)
    if n < 5:
        return None, None, None

    best_score, best_square = float('inf'), None
    for combo in combinations(range(n), 4):
        sc = square_score([points[i] for i in combo])
        if sc < best_score:
            best_score, best_square = sc, combo

    if best_square is None or best_score > score_threshold:
        return None, None, best_score

    best_square = tuple(sorted(best_square))
    remaining = [i for i in range(n) if i not in best_square]

    star = remaining[0]
    if len(remaining) > 1:
        best_var = float('inf')
        for k in remaining:
            dk = [np.hypot(points[k][0] - points[i][0],
                           points[k][1] - points[i][1]) for i in best_square]
            var = np.std(dk) / (np.mean(dk) + 1e-6)
            if var < best_var:
                best_var, star = var, k
    return best_square, star, best_score


Arm = Arm_Device()
time.sleep(.1)

try:
    Arm.Arm_serial_servo_write6(90, 90, 40, -40, 90, 90, 1500)
    time.sleep(4)

    cap = cv2.VideoCapture(1)

    center_mass_point = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bg   = cv2.GaussianBlur(gray, (0, 0), 30)
        norm = cv2.divide(gray, bg, scale=255)
        norm_blur = cv2.medianBlur(norm, 5)

        mask_for_input_cicle = cv2.inRange(gray, 21, 100)

        mask_for_input_cicle = cv2.medianBlur(mask_for_input_cicle, 5)

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_for_input_cicle = cv2.morphologyEx(mask_for_input_cicle, cv2.MORPH_CLOSE, k_close, iterations=2)

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        mask_for_input_cicle = cv2.morphologyEx(mask_for_input_cicle, cv2.MORPH_OPEN, k_open, iterations=1)

        kernel = np.ones((5, 5), np.uint8)
        mask_for_input_cicle = cv2.morphologyEx(mask_for_input_cicle, cv2.MORPH_OPEN, kernel)
        mask_for_input_cicle = cv2.morphologyEx(mask_for_input_cicle, cv2.MORPH_CLOSE, kernel)
        
        th = cv2.adaptiveThreshold(norm, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 50)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  k)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)

        param2, minR, maxR, fill_t = 21, 7, 24, 0.2

        circles = cv2.HoughCircles(
            norm_blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=max(20, minR * 2),
            param1=100, param2=max(1, param2),
            minRadius=minR, maxRadius=maxR
        )

        detected = []
        if circles is not None:
            for x, y, r in np.round(circles[0]).astype(int):
                f = circle_fill(th, x, y, r)
                if f >= fill_t:
                    detected.append((x, y, r, f))

        contours, hierarchy = cv2.findContours(mask_for_input_cicle, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

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
            center_mass_point = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.circle(frame, (center_mass_point[0], center_mass_point[1]), 5, (0, 0, 255), -1)
            cv2.circle(frame, (center_mass_point[0], center_mass_point[1]), 3, (255, 255, 255), -1)
        
        if len(detected) == 4:
            sc = square_score([p[:2] for p in detected])

            if sc < 0.35:
                points = [i[:2] for i in detected]
                center_ponint_coordinate = drawing_middle_of_square(points)
                if center_mass_point is not None:
                    draw_line_beetween_two_centers(center_mass_point, center_ponint_coordinate)
            else:
                for x, y, r, _ in detected:
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.putText(frame, "No STAR (only 4)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        elif len(detected) >= 5:
            pts = [(p[0], p[1]) for p in detected]
            sq_idx, star_idx, score = find_star(pts, score_threshold=0.35)

            if sq_idx is not None:
                for i in sq_idx:
                    x, y, r, _ = detected[i]
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(frame, (x, y), 2, (0, 200, 0), -1)

                x, y, r, _ = detected[star_idx]
                cv2.circle(frame, (x, y), r + 5, (0, 0, 255), 3)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                cv2.line(frame, (x - 12, y), (x + 12, y), (0, 0, 255), 2)
                cv2.line(frame, (x, y - 12), (x, y + 12), (0, 0, 255), 2)
                cv2.putText(frame, "STAR", (x + r + 8, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"({x},{y})", (x + r + 8, y + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                print(f"square={sq_idx}  star={star_idx}  "
                      f"coord=({x},{y})  score={score:.3f}")

                points_sq = [detected[i][:2] for i in sq_idx]
                center_point_coordinate = drawing_middle_of_square(points_sq)

                if center_mass_point is not None:
                    draw_line_beetween_two_centers(center_mass_point, center_point_coordinate)

            else:
                for x, y, r, _ in detected:
                    cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
                cv2.putText(frame, "Square not detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                print(f"no square. n={len(detected)} score={score}")

            cv2.drawContours(frame, contours, inner_idx, (0, 255, 0), 1)
            
        else:
            for x, y, r, _ in detected:
                cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
            print(f"too few: {len(detected)}")


        cv2.imshow("frame", frame)
        cv2.imshow("mask",  th)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    del Arm