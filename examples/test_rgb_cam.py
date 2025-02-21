from hardware.camera import RGBCamera
import cv2
cam = RGBCamera(device_id=0, width=640, height=480, fps=6)
cam.connect()
cam.plot_image_bundle()