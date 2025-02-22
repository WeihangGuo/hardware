import pyrealsense2 as rs
context = rs.context()

# Enumerate all connected devices
devices = context.query_devices()

# Check the device list and choose an appropriate device ID
for i, device in enumerate(devices):
    print(f"Device {i}: {device.get_info(rs.camera_info.serial_number)}")