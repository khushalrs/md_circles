# multi_drone_circles/util.py
import math

def quaternion_from_yaw(yaw):
    cy = math.cos(yaw*0.5)
    sy = math.sin(yaw*0.5)
    return (0.0, 0.0, sy, cy)
