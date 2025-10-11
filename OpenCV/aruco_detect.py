import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
params = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)

# dummy camera calibration values
mtx = np.array([[800, 0, 320],
                [0, 800, 240],
                [0, 0, 1]], dtype=float)
dist = np.zeros((5, 1))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=params)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.03)
            pos = f"x:{tvec[0][0]:.2f} y:{tvec[0][1]:.2f} z:{tvec[0][2]:.2f}"
            cv2.putText(frame, pos, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

