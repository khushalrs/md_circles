#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection3DArray

class CirclesMarkers(Node):
    def __init__(self):
        super().__init__('circles_markers')
        self.declare_parameter('merged_topic', '/circles/merged')
        self.declare_parameter('markers_topic', '/circles/markers')
        self.declare_parameter('labels_topic',  '/circles/labels')
        self.declare_parameter('global_frame',  'map')

        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value
        merged_topic = self.get_parameter('merged_topic').get_parameter_value().string_value
        markers_topic = self.get_parameter('markers_topic').get_parameter_value().string_value
        labels_topic  = self.get_parameter('labels_topic').get_parameter_value().string_value

        self.sub = self.create_subscription(Detection3DArray, merged_topic, self.cb, 10)
        self.pub_markers = self.create_publisher(MarkerArray, markers_topic, 10)
        self.pub_labels  = self.create_publisher(MarkerArray, labels_topic, 10)

    def cb(self, msg: Detection3DArray):
        spheres = MarkerArray()
        labels  = MarkerArray()

        stamp = self.get_clock().now().to_msg()
        for i, det in enumerate(msg.detections):
            if not det.results: 
                continue
            p = det.results[0].pose.pose.position
            # radius is stored as diameter in bbox.size.{x,y,z}
            r = 0.5 * max(det.bbox.size.x, det.bbox.size.y, det.bbox.size.z)
            # Sphere marker
            m = Marker()
            m.header.frame_id = self.global_frame
            m.header.stamp = stamp
            m.ns = 'circles'
            try:
                id_int = int(det.id) if det.id else i
            except Exception:
                id_int = i
            m.id = id_int
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = p.x; m.pose.position.y = p.y; m.pose.position.z = p.z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = r*2.0
            m.color.a = 0.6    # visible transparency
            m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0
            m.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
            spheres.markers.append(m)

            # Text label (radius in meters)
            t = Marker()
            t.header.frame_id = self.global_frame
            t.header.stamp = stamp
            t.ns = 'labels'
            t.id = id_int
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = p.x; t.pose.position.y = p.y; t.pose.position.z = p.z + r + 0.05
            t.pose.orientation.w = 1.0
            t.scale.z = max(0.05, 0.25*r)   # text height
            t.color.a = 1.0
            t.color.r = 1.0; t.color.g = 1.0; t.color.b = 1.0
            id_str = det.id if det.id else str(id_int)
            t.text = f"id={id_str} (x={p.x:.2f}, y={p.y:.2f}, z={p.z:.2f})"
            t.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
            labels.markers.append(t)

        self.pub_markers.publish(spheres)
        self.pub_labels.publish(labels)

def main():
    rclpy.init()
    n = CirclesMarkers()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
