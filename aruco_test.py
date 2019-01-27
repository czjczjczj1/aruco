
import matplotlib.pyplot as plt
import numpy as np
import cv2
import cv2.aruco as aruco

mtx = np.array([[2946.48,0,1980.53],[0, 2945.41, 1129.25],[0,0,1],])
dist = np.array([0.226317, -1.21478, 0.00170689, -0.000334551, 1.9892])

cap = cv2.VideoCapture(1)
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    frame_detected_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.042, mtx, dist)

    for i in range(len(tvecs)):
       frame_axis_image = cv2.aruco.drawAxis(frame_detected_markers.copy(), mtx, dist, rvecs[i], tvecs[i], 0.05)

    if ids is not None:
        for i in range(len(ids)):
            c = corners[i][0]
            plt.plot([c[:, 0].mean()], [c[:, 1].mean()], "+", label="id={0}".format(ids[i]))

    # Display the resulting frame
    cv2.imshow('frame', frame_axis_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()