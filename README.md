# Hardware 
A Python package for interfacing with cameras and CUDA-enabled GPUs.

This package is based on [hardware](https://github.com/skumra/robotic-grasping/tree/master/hardware)

## Installation
```bash
pip install -e . 
```

As of February 2025, `pyrealsense2` has pre-built wheels from Python `3.7` to `3.11`. 
It supports Ubuntu 16, 18, 20, and 22, as well as kernel versions `<= 5.15`.

## Usage
Calibrate the rgb camera using `python -m hardware.calibrate_rgb_cam`