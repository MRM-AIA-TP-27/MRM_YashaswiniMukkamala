import cv2
import numpy as np

cap = cv2.VideoCapture(0)

low_y = np.array([18, 90, 90])
high_y = np.array([45, 255, 255])

while True:
    ret, f = cap.read()
    if not ret:
        break

    f = cv2.resize(f, (640, 480))
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (11, 11), 0)
    mask = cv2.inRange(blur, low_y, high_y)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    conts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in conts:
        if cv2.contourArea(c) < 500:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        if r > 10:
            cv2.circle(f, (int(x), int(y)), int(r), (0,255,255), 3)
            cv2.circle(f, center, 5, (0,0,255), -1)
            cv2.putText(f, "Tennis Ball", (int(x - r), int(y - r - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("mask", mask)
    cv2.imshow("Tennis Ball Detection", f)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
