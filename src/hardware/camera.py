import logging

import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
import cv2
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class AbstractCamera(ABC):
    def __init__(self, device_id, width=640, height=480, fps=6):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

    @abstractmethod
    def connect(self):
        """Establish a connection with the camera."""
        pass

    @abstractmethod
    def get_image_bundle(self):
        """Capture and return an image bundle."""
        pass

    @abstractmethod
    def plot_image_bundle(self):
        """Display the image bundle."""
        pass

class RGBCamera(AbstractCamera):
    @dataclass
    class Intrinsics:
        fx: float
        fy: float
        ppx: float
        ppy: float
    
    def __init__(self, device_id, intrinsics=None, width=640, height=480, fps=6):
        super().__init__(device_id, width, height, fps)
        self.device = None
        self.intrinsics = intrinsics # Intrinsics should be a dictionary with keys: fx, fy, ppx, ppy

    def connect(self):
        self.device = cv2.VideoCapture(self.device_id)
        if not self.device.isOpened():
            raise Exception(f"Could not open video device {self.device_id}")
        
        self.device.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.device.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.device.set(cv2.CAP_PROP_FPS, self.fps)
    
    def get_image_bundle(self):
        ret, frame = self.device.read()
        if not ret:
            raise Exception("Failed to capture frame from camera")
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Return the RGB image
        return {
            'rgb': rgb_image, 
            'aligned_depth': None,  # No depth image for RGB camera
        }
    
    def plot_image_bundle(self):
        images = self.get_image_bundle()
        rgb = images['rgb']
        plt.imshow(rgb)
        plt.title('RGB Image')
        plt.axis('off')
        plt.show()

class RealSenseCamera(AbstractCamera):
    def __init__(self,
                 device_id,
                 width=640,
                 height=480,
                 fps=6):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.scale = None
        self.intrinsics = None

    def connect(self):
        # Start and configure
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(self.device_id))
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()

    def get_image_bundle(self):
        frames = self.pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.first(rs.stream.color)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image *= self.scale
        color_image = np.asanyarray(color_frame.get_data())

        depth_image = np.expand_dims(depth_image, axis=2)

        return {
            'rgb': color_image,
            'aligned_depth': depth_image,
        }

    def plot_image_bundle(self):
        images = self.get_image_bundle()

        rgb = images['rgb']
        depth = images['aligned_depth']

        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')

        plt.show()


if __name__ == '__main__':
    cam = RealSenseCamera(device_id=830112070066)
    cam.connect()
    while True:
        cam.plot_image_bundle()