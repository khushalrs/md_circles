# multi_drone_circles/circle_detector_node.py
import rclpy, cv2, numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import Header, String
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import tf2_ros
import tf_transformations as tft
from rclpy.time import Time
from collections import deque


def _fit_circle_2d(points2d: np.ndarray):
    if points2d.shape[0] < 3:
        raise ValueError('Need at least three points for circle fit')

    x = points2d[:, 0]
    y = points2d[:, 1]
    A = np.column_stack((x, y, np.ones_like(x)))
    b = -(x ** 2 + y ** 2)

    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f'Least squares failed: {exc}') from exc

    B, C, D = sol
    cx = -0.5 * B
    cy = -0.5 * C
    r_sq = cx * cx + cy * cy - D
    if r_sq <= 0.0:
        raise ValueError('Computed negative radius squared')

    return np.array([cx, cy]), float(np.sqrt(r_sq))


class CircleDetector(Node):
    def __init__(self):
        super().__init__('circle_detector')
        self.bridge = CvBridge()
        self.declare_parameter('rgb_topic', '/drone1/rgb_camera')
        self.declare_parameter('depth_topic', '/drone1/depth_camera')
        self.declare_parameter('camera_info_topic', '/drone1/rgb_camera/camera_info')
        self.declare_parameter('camera_frame', 'drone1/camera_link')      # your camera optical or camera_link
        self.declare_parameter('use_msg_frame', True)  # prefer RGB header.frame_id if present
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('drone_id', 'drone1')
        self.declare_parameter('min_radius_px', 8)
        self.declare_parameter('max_radius_px', 300)
        self.declare_parameter('dp', 1.2)
        self.declare_parameter('canny_thresh', 100)
        self.declare_parameter('acc_thresh', 30)   # Hough accumulator threshold
        self.declare_parameter('depth_scale', 1.0) # if depth is in meters, keep 1.0; if in mm, use 0.001
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('min_contour_area', 200)
        self.declare_parameter('dedupe_threshold_m', 0.3)
        self.declare_parameter('track_timeout_sec', 4.0)
        self.declare_parameter('track_history_length', 48)
        # Quality gates
        self.declare_parameter('rim_width_px', 3)
        self.declare_parameter('min_depth_coverage', 0.3)
        self.declare_parameter('max_plane_thickness_m', 0.05)
        self.declare_parameter('max_circle_rms_m', 0.05)
        self.declare_parameter('min_radius_m', 0.005)
        self.declare_parameter('max_radius_m', 1.0)
        self.declare_parameter('max_reproj_err_px', 6.0)
        self.declare_parameter('min_view_abs_dot', 0.05)
        self.declare_parameter('enable_geometry_gating', True)
        self.declare_parameter('use_red_mask', True)

        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.use_msg_frame = bool(self.get_parameter('use_msg_frame').get_parameter_value().bool_value)
        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value
        self.drone_id = self.get_parameter('drone_id').get_parameter_value().string_value
        self.min_r = int(self.get_parameter('min_radius_px').get_parameter_value().integer_value or 8)
        self.max_r = int(self.get_parameter('max_radius_px').get_parameter_value().integer_value or 300)
        self.dp = float(self.get_parameter('dp').get_parameter_value().double_value or 1.2)
        self.canny = int(self.get_parameter('canny_thresh').get_parameter_value().integer_value or 100)
        self.acc = int(self.get_parameter('acc_thresh').get_parameter_value().integer_value or 30)
        self.depth_scale = float(self.get_parameter('depth_scale').get_parameter_value().double_value or 1.0)
        self.publish_debug = bool(self.get_parameter('publish_debug').get_parameter_value().bool_value)
        self.min_contour_area = float(self.get_parameter('min_contour_area').get_parameter_value().double_value or 200.0)
        self.dedupe_threshold = float(self.get_parameter('dedupe_threshold_m').get_parameter_value().double_value or 0.3)
        self.track_timeout = float(self.get_parameter('track_timeout_sec').get_parameter_value().double_value or -1.0)
        self.track_history_length = max(1, int(self.get_parameter('track_history_length').get_parameter_value().integer_value or 48))
        # Gates
        self.rim_width_px = int(self.get_parameter('rim_width_px').get_parameter_value().integer_value or 3)
        self.min_depth_coverage = float(self.get_parameter('min_depth_coverage').get_parameter_value().double_value or 0.6)
        self.max_plane_thickness_m = float(self.get_parameter('max_plane_thickness_m').get_parameter_value().double_value or 0.02)
        self.max_circle_rms_m = float(self.get_parameter('max_circle_rms_m').get_parameter_value().double_value or 0.02)
        self.min_radius_m = float(self.get_parameter('min_radius_m').get_parameter_value().double_value or 0.01)
        self.max_radius_m = float(self.get_parameter('max_radius_m').get_parameter_value().double_value or 0.5)
        self.max_reproj_err_px = float(self.get_parameter('max_reproj_err_px').get_parameter_value().double_value or 6.0)
        self.min_view_abs_dot = float(self.get_parameter('min_view_abs_dot').get_parameter_value().double_value or 0.05)
        self.enable_geometry_gating = bool(self.get_parameter('enable_geometry_gating').get_parameter_value().bool_value)
        self.use_red_mask = bool(self.get_parameter('use_red_mask').get_parameter_value().bool_value)

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=10)

        self.rgb_sub = Subscriber(self, Image, self.rgb_topic, qos_profile=qos)
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos)
        self.info_sub = self.create_subscription(CameraInfo, self.info_topic, self.info_cb, 10)

        self.K = None  # fx, fy, cx, cy
        self.last_info = None
        self.has_warned_no_k = False

        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.cb)

        self.pub = self.create_publisher(Detection3DArray, 'detections_3d', 10)
        self.meta_pub = self.create_publisher(String, 'detector_meta', 10)  # optional

        if self.publish_debug:
            self.debug_pub = self.create_publisher(Image, 'debug/circles', 10)

        self.tracked_circles = []  # list of dict with keys: center_map(np.array), radius(float), stamp(Time msg), time_wall(Time)

        self.get_logger().info(f'[{self.drone_id}] CircleDetector up. RGB={self.rgb_topic}, DEPTH={self.depth_topic}')

    def info_cb(self, msg: CameraInfo):
        self.last_info = msg
        fx = msg.k[0]; fy = msg.k[4]; cx = msg.k[2]; cy = msg.k[5]
        self.K = (fx, fy, cx, cy)

    def cb(self, rgb_msg: Image, depth_msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'CV bridge RGB error: {e}')
            return

        dbg = rgb.copy() if self.publish_debug else None

        detections_world = []

        # Only proceed with full detection if we have camera info
        if self.K is not None:
            self.has_warned_no_k = False # Reset warning flag if we get info again
            try:
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32) * self.depth_scale
                fx, fy, cx, cy = self.K

                # Build binary mask either from color or edges
                kernel = np.ones((3, 3), np.uint8)
                if self.use_red_mask:
                    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
                    lower_red1 = np.array([0, 120, 70])
                    upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([170, 120, 70])
                    upper_red2 = np.array([180, 255, 255])
                    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                    base_mask = cv2.bitwise_or(mask1, mask2)
                else:
                    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, self.canny, self.canny*2)
                    base_mask = cv2.dilate(edges, kernel, iterations=1)

                mask_clean = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

                contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                frame_centers_map = []

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < self.min_contour_area:
                        continue

                    _, r_px_est = cv2.minEnclosingCircle(contour)
                    if r_px_est < self.min_r or r_px_est > self.max_r:
                        continue

                    # Build ring (rim) by eroding the filled mask
                    mask_filled = np.zeros_like(mask_clean, dtype=np.uint8)
                    cv2.drawContours(mask_filled, [contour], -1, 255, thickness=-1)
                    if self.publish_debug:
                        try:
                            cv2.drawContours(dbg, [contour], -1, (255, 165, 0), 1)
                        except Exception:
                            pass
                    if self.rim_width_px > 0:
                        inner = cv2.erode(mask_filled, np.ones((3,3), np.uint8), iterations=max(1, self.rim_width_px))
                        rim = cv2.subtract(mask_filled, inner)
                    else:
                        rim = mask_filled

                    ys, xs = np.nonzero(rim)
                    if xs.size < 12:
                        continue

                    depth_vals = depth[ys, xs]
                    valid_mask = np.isfinite(depth_vals) & (depth_vals > 0.05)
                    coverage = float(np.count_nonzero(valid_mask)) / float(xs.size)
                    if coverage < self.min_depth_coverage or np.count_nonzero(valid_mask) < 12:
                        continue

                    xs = xs[valid_mask]
                    ys = ys[valid_mask]
                    depth_vals = depth_vals[valid_mask]

                    X = (xs - cx) / fx * depth_vals
                    Y = (ys - cy) / fy * depth_vals
                    Z = depth_vals
                    pts_cam = np.vstack((X, Y, Z)).T

                    # Prepare fallback points from filled mask
                    ys_f, xs_f = np.nonzero(mask_filled)
                    depth_full = depth[ys_f, xs_f]
                    valid_full = np.isfinite(depth_full) & (depth_full > 0.05)
                    xs_f = xs_f[valid_full]; ys_f = ys_f[valid_full]; depth_full = depth_full[valid_full]
                    Xf = (xs_f - cx) / fx * depth_full
                    Yf = (ys_f - cy) / fy * depth_full
                    Zf = depth_full
                    pts_cam_full = np.vstack((Xf, Yf, Zf)).T

                    use_fallback = not self.enable_geometry_gating
                    center_cam = None
                    radius_m = None

                    if self.enable_geometry_gating:
                        # Estimate local plane using PCA and check thickness and view angle
                        centroid = pts_cam.mean(axis=0)
                        pts_centered = pts_cam - centroid
                        try:
                            _, s, vt = np.linalg.svd(pts_centered, full_matrices=False)
                        except np.linalg.LinAlgError:
                            use_fallback = True
                        else:
                            n = vt[2]
                            plane_thickness = float(s[2]) / max(1.0, np.sqrt(pts_cam.shape[0]))
                            view_abs_dot = abs(float(n[2]))
                            # Gating
                            if (not np.isfinite(plane_thickness) or plane_thickness > self.max_plane_thickness_m or
                                view_abs_dot < self.min_view_abs_dot):
                                use_fallback = True
                            else:
                                # Fit circle in plane coordinates
                                u_axis = vt[0]
                                v_axis = vt[1]
                                coords_u = pts_centered @ u_axis
                                coords_v = pts_centered @ v_axis
                                pts2d = np.column_stack((coords_u, coords_v))
                                try:
                                    center_2d, radius_val = _fit_circle_2d(pts2d)
                                except ValueError as e:
                                    self.get_logger().debug(f'Circle fit failed: {e}')
                                    use_fallback = True
                                else:
                                    diffs = pts2d - center_2d
                                    rr = np.linalg.norm(diffs, axis=1)
                                    rms = float(np.sqrt(np.mean((rr - radius_val) ** 2))) if rr.size > 0 else np.inf
                                    if (not np.isfinite(radius_val) or radius_val <= 0.0 or
                                        radius_val < self.min_radius_m or radius_val > self.max_radius_m or
                                        not np.isfinite(rms) or rms > self.max_circle_rms_m):
                                        use_fallback = True
                                    else:
                                        center_cam = centroid + center_2d[0] * u_axis + center_2d[1] * v_axis
                                        radius_m = float(radius_val)

                    if use_fallback:
                        # Fall back to unconstrained 3D circle fit
                        try:
                            center_cam_f, radius_f = self._fit_circle_from_points(pts_cam_full)
                            center_cam = center_cam_f
                            radius_m = float(radius_f)
                        except Exception:
                            continue

                    # Draw debug overlay in image space regardless of TF
                    if self.publish_debug:
                        try:
                            proj_u_dbg = int(np.clip((center_cam[0] / max(1e-6, center_cam[2])) * fx + cx, 0, rgb.shape[1]-1))
                            proj_v_dbg = int(np.clip((center_cam[1] / max(1e-6, center_cam[2])) * fy + cy, 0, rgb.shape[0]-1))
                            proj_r_dbg = int(max(1.0, radius_m * fx / max(1e-6, center_cam[2])))
                            cv2.circle(dbg, (proj_u_dbg, proj_v_dbg), proj_r_dbg, (0, 200, 0), 2)
                            cv2.circle(dbg, (proj_u_dbg, proj_v_dbg), 2, (0, 0, 255), 2)
                        except Exception:
                            pass

                    try:
                        src_frame = rgb_msg.header.frame_id if (self.use_msg_frame and rgb_msg.header.frame_id) else self.camera_frame
                        t = self.tf_buffer.lookup_transform(self.global_frame, src_frame,
                                                            rgb_msg.header.stamp, rclpy.duration.Duration(seconds=0.05))
                        tx = t.transform.translation.x
                        ty = t.transform.translation.y
                        tz = t.transform.translation.z
                        q = t.transform.rotation
                        R = tft.quaternion_matrix([q.x, q.y, q.z, q.w])[0:3,0:3]
                        p_map = R @ center_cam + np.array([tx, ty, tz])

                        p_map = p_map.astype(np.float64)

                        duplicate = False
                        for prev in frame_centers_map:
                            if np.linalg.norm(prev - p_map) < self.dedupe_threshold:
                                duplicate = True
                                break
                        if duplicate:
                            continue

                        frame_centers_map.append(p_map.copy())

                        # Reprojection check: compare to 2D centroid of mask
                        proj_u = (center_cam[0] / max(1e-6, center_cam[2])) * fx + cx
                        proj_v = (center_cam[1] / max(1e-6, center_cam[2])) * fy + cy
                        M = cv2.moments(mask_filled, binaryImage=True)
                        if M["m00"] > 0:
                            cx_img = M["m10"]/M["m00"]
                            cy_img = M["m01"]/M["m00"]
                            reproj_err = float(np.hypot(proj_u - cx_img, proj_v - cy_img))
                        else:
                            reproj_err = 0.0
                        if reproj_err > self.max_reproj_err_px:
                            continue

                        detections_world.append((p_map, radius_m, rgb_msg.header.stamp))

                        if self.publish_debug:
                            try:
                                proj_u_i = int(np.clip(proj_u, 0, rgb.shape[1]-1))
                                proj_v_i = int(np.clip(proj_v, 0, rgb.shape[0]-1))
                                proj_r_px = int(max(1.0, radius_m * fx / max(1e-6, center_cam[2])))
                                cv2.circle(dbg, (proj_u_i, proj_v_i), proj_r_px, (0, 255, 0), 2)
                                cv2.circle(dbg, (proj_u_i, proj_v_i), 2, (0, 0, 255), 2)
                            except Exception:
                                pass

                    except Exception as e:
                        self.get_logger().warn(f'TF lookup/apply failed: {e}')
                        continue
            except Exception as e:
                self.get_logger().warn(f'Circle detection processing failed: {e}')
        else:
            if not self.has_warned_no_k:
                self.get_logger().warn('No camera info received, skipping circle detection. This message will not repeat.')
                self.has_warned_no_k = True

        # Publish debug image if enabled. It will be raw if processing was skipped/failed.
        if self.publish_debug and dbg is not None:
            try:
                self.debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))
            except Exception as e:
                self.get_logger().warn(f"Failed to publish debug image: {e}")

        if self.K is not None:
            self._update_tracks(detections_world)

        else:
            # Even without camera intrinsics we still prune stale tracks
            self._prune_tracks(self.get_clock().now())
            self._merge_tracks()

        out = Detection3DArray()
        out.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.global_frame)

        for track in self.tracked_circles:
            det = Detection3D()
            det.header = Header(stamp=track['stamp'], frame_id=self.global_frame)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = "circle"
            hyp.hypothesis.score = 1.0
            pose_map = Pose()
            pose_map.position.x = float(track['center_map'][0])
            pose_map.position.y = float(track['center_map'][1])
            pose_map.position.z = float(track['center_map'][2])
            pose_map.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            hyp.pose.pose = pose_map
            det.results.append(hyp)

            det.bbox.size.x = track['radius'] * 2.0
            det.bbox.size.y = track['radius'] * 2.0
            det.bbox.size.z = track['radius'] * 2.0

            out.detections.append(det)

        # Publish detections and metadata
        self.pub.publish(out)
        self.meta_pub.publish(String(data=self.drone_id))

    def _update_tracks(self, detections_world):
        now_wall = self.get_clock().now()

        for p_map, radius_m, stamp_msg in detections_world:
            p_map = np.asarray(p_map, dtype=np.float64)
            matched = None
            for track in self.tracked_circles:
                if np.linalg.norm(track['center_map'] - p_map) < self.dedupe_threshold:
                    matched = track
                    break

            if matched is not None:
                matched['history_center'].append(p_map)
                matched['history_radius'].append(float(radius_m))
                matched['center_map'] = np.mean(np.vstack(matched['history_center']), axis=0)
                matched['radius'] = float(np.mean(matched['history_radius']))
                matched['stamp'] = stamp_msg
                matched['time_wall'] = now_wall
            else:
                self.tracked_circles.append({
                    'center_map': p_map,
                    'radius': float(radius_m),
                    'stamp': stamp_msg,
                    'time_wall': now_wall,
                    'history_center': deque([p_map], maxlen=self.track_history_length),
                    'history_radius': deque([float(radius_m)], maxlen=self.track_history_length)
                })

        self._prune_tracks(now_wall)
        self._merge_tracks()

    def _prune_tracks(self, reference_time: Time):
        if self.track_timeout <= 0.0:
            return

        threshold_ns = self.track_timeout * 1e9
        kept = []
        for track in self.tracked_circles:
            age = (reference_time - track['time_wall']).nanoseconds
            if age < 0 or age <= threshold_ns:
                kept.append(track)
        self.tracked_circles = kept

    def _merge_tracks(self):
        if not self.tracked_circles:
            return

        merged = []
        for track in sorted(self.tracked_circles, key=lambda t: t['time_wall'].nanoseconds, reverse=True):
            keep = True
            for existing in merged:
                if np.linalg.norm(existing['center_map'] - track['center_map']) < self.dedupe_threshold:
                    # keep the fresher entry (merged already sorted by time desc)
                    keep = False
                    break
            if keep:
                merged.append(track)

        self.tracked_circles = merged

    def _fit_circle_from_points(self, pts_cam: np.ndarray):
        if pts_cam.shape[0] < 3:
            raise ValueError('Insufficient points for circle fit')

        centroid = pts_cam.mean(axis=0)
        pts_centered = pts_cam - centroid

        try:
            _, _, vt = np.linalg.svd(pts_centered, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f'SVD failed: {exc}') from exc

        u_axis = vt[0]
        v_axis = vt[1]

        coords_u = pts_centered @ u_axis
        coords_v = pts_centered @ v_axis
        pts2d = np.column_stack((coords_u, coords_v))

        center_2d, radius = _fit_circle_2d(pts2d)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError('Invalid radius from fit')

        center_cam = centroid + center_2d[0] * u_axis + center_2d[1] * v_axis
        return center_cam, float(radius)

def main():
    rclpy.init()
    node = CircleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
