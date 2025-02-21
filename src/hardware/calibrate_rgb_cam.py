import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# get chessboard corners size, width and height
parser = argparse.ArgumentParser(description='Camera Calibration')
parser.add_argument('--width', type=int, default=7, help='Width of the chessboard, inner corners')
parser.add_argument('--height', type=int, default=5, help='Height of the chessboard, inner corners')
parser.add_argument('--device', type=int, default=0, help='Camera device ID, default is 0')
args = parser.parse_args()
# Set up the chessboard size
chessboard_size = (args.width, args.height)

# Prepare object points (3D coordinates in the real world)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Open camera
cap = cv2.VideoCapture(0)  # 0 for default webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize the frame to reduce processing time
    cv2.imshow("Chessboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and show corners
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
        cv2.imshow("Chessboard", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Calibrate camera and get intrinsics
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Display intrinsics
# print("Camera Matrix (Intrinsics):\n", camera_matrix)
print(f"fx: {camera_matrix[0, 0]}, fy: {camera_matrix[1, 1]}, ppx: {camera_matrix[0, 2]}, ppy: {camera_matrix[1, 2]}")
print("Distortion Coefficients:\n", dist_coeffs)
