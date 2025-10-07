# multi_drone_circles/dedupe_and_logger_node.py
import rclpy, numpy as np, csv, os
from rclpy.node import Node
from std_msgs.msg import String, Header
from vision_msgs.msg import Detection3DArray, Detection3D
from rclpy.qos import QoSProfile
from rclpy.time import Time


class Track:
    def __init__(self, x, P, t: Time, track_id: int, radius_init: float | None = None):
        # state: [x y z vx vy vz]^T
        self.x = x.astype(np.float64).reshape(6,1)
        self.P = P.astype(np.float64)
        self.last_update = t
        self.id = track_id
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.radius = float(radius_init) if (radius_init is not None and np.isfinite(radius_init)) else float('nan')

    def predict(self, t: Time, q_pos2: float, q_vel2: float):
        dt = max(0.0, (t - self.last_update).nanoseconds * 1e-9)
        if dt == 0.0:
            return
        F = np.eye(6)
        F[0,3] = dt; F[1,4] = dt; F[2,5] = dt
        G = np.array([[0.5*dt*dt, 0, 0],
                      [0, 0.5*dt*dt, 0],
                      [0, 0, 0.5*dt*dt],
                      [dt, 0, 0],
                      [0, dt, 0],
                      [0, 0, dt]], dtype=np.float64)
        Q = np.diag([q_pos2, q_pos2, q_pos2, q_vel2, q_vel2, q_vel2])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_update = t

    def innovation(self, z: np.ndarray, R: np.ndarray):
        H = np.zeros((3,6)); H[0,0] = 1; H[1,1] = 1; H[2,2] = 1
        z = z.reshape(3,1)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        return y, S, H

    def update(self, z: np.ndarray, R: np.ndarray):
        y, S, H = self.innovation(z, R)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        K = self.P @ H.T @ S_inv
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P
        self.hits += 1
        self.misses = 0

class DedupeAndLogger(Node):
    def __init__(self):
        super().__init__('dedupe_and_logger')
        self.declare_parameter('global_frame', 'map')
        # EKF + association params
        self.declare_parameter('alpha_geom', 1.0)      # weight for geometry (no descriptor yet)
        self.declare_parameter('gate_chi2', 30.0)      # loosen gate for noisy depth/pose
        self.declare_parameter('R_diag_m', [0.08, 0.08, 0.08])  # per-axis meas std (m)
        self.declare_parameter('q_pos2', 1e-4)         # process noise pos^2
        self.declare_parameter('q_vel2', 1e-3)         # process noise vel^2
        self.declare_parameter('confirm_hits', 2)      # promote after K hits
        self.declare_parameter('max_misses', 10)       # delete after misses
        self.declare_parameter('merge_distance_m', 0.8)
        self.declare_parameter('radius_alpha', 0.4)    # EMA for radius per matched update
        self.declare_parameter('default_radius_m', 0.15)
        self.declare_parameter('fuse_distance_m', 0.8) # allow second sensor update to same track
        self.declare_parameter('track_merge_m', 0.8)   # periodic hard-merge for near-duplicates
        self.declare_parameter('log_dir', 'circle_logs')
        self.declare_parameter('csv_name', 'circles_merged.csv')

        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value
        self.alpha_geom = float(self.get_parameter('alpha_geom').get_parameter_value().double_value or 1.0)
        R_list = self.get_parameter('R_diag_m').get_parameter_value().double_array_value
        if not R_list:
            R_list = [0.02, 0.02, 0.02]
        self.R = np.diag(np.array(R_list, dtype=np.float64)**2)
        self.q_pos2 = float(self.get_parameter('q_pos2').get_parameter_value().double_value or 1e-4)
        self.q_vel2 = float(self.get_parameter('q_vel2').get_parameter_value().double_value or 1e-3)
        self.gate_chi2 = float(self.get_parameter('gate_chi2').get_parameter_value().double_value or 16.27)
        self.confirm_hits = int(self.get_parameter('confirm_hits').get_parameter_value().integer_value or 2)
        self.max_misses = int(self.get_parameter('max_misses').get_parameter_value().integer_value or 10)
        self.merge_distance = float(self.get_parameter('merge_distance_m').get_parameter_value().double_value or 0.8)
        self.radius_alpha = float(self.get_parameter('radius_alpha').get_parameter_value().double_value or 0.4)
        self.default_radius = float(self.get_parameter('default_radius_m').get_parameter_value().double_value or 0.15)
        self.fuse_distance = float(self.get_parameter('fuse_distance_m').get_parameter_value().double_value or 0.8)
        self.track_merge_m = float(self.get_parameter('track_merge_m').get_parameter_value().double_value or 0.8)

        log_dir = self.get_parameter('log_dir').get_parameter_value().string_value
        csv_name = self.get_parameter('csv_name').get_parameter_value().string_value
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, csv_name)
        self._init_csv()

        qos = QoSProfile(depth=10)
        self.meta_drone1 = None
        self.meta_drone2 = None

        self.sub1 = self.create_subscription(Detection3DArray, '/drone1/detections_3d', self.cb1, qos)
        self.sub2 = self.create_subscription(Detection3DArray, '/drone2/detections_3d', self.cb2, qos)
        self.m1 = self.create_subscription(String, '/drone1/detector_meta', self.mcb1, 10)
        self.m2 = self.create_subscription(String, '/drone2/detector_meta', self.mcb2, 10)

        self.pub = self.create_publisher(Detection3DArray, '/circles/merged', 10)

        self.buf1 = None
        self.buf2 = None
        # Global tracks
        self.tracks: list[Track] = []
        self._next_id = 1

        self.timer = self.create_timer(0.1, self.tick)  # 10 Hz merge

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['stamp_sec', 'stamp_nsec', 'x', 'y', 'z', 'radius_m', 'cluster_size', 'drone_ids'])

    def mcb1(self, msg: String): self.meta_drone1 = msg.data or 'drone1'
    def mcb2(self, msg: String): self.meta_drone2 = msg.data or 'drone2'

    def cb1(self, msg: Detection3DArray): self.buf1 = (msg, self.meta_drone1 or 'drone1')
    def cb2(self, msg: Detection3DArray): self.buf2 = (msg, self.meta_drone2 or 'drone2')

    def tick(self):
        if self.buf1 is None and self.buf2 is None:
            return

        # Gather candidates from both drones
        candidates = []  # list of (z: np.ndarray, r: float, drone_id: str)
        stamp_candidates = []
        for buf in [self.buf1, self.buf2]:
            if buf is None:
                continue
            msg, drone_id = buf
            for d in msg.detections:
                if not d.results:
                    continue
                p = d.results[0].pose.pose.position
                r = 0.5 * max(d.bbox.size.x, d.bbox.size.y, d.bbox.size.z)
                candidates.append((np.array([p.x, p.y, p.z], dtype=np.float64), float(r), drone_id))
                stamp_candidates.append(Time.from_msg(msg.header.stamp))

        # Clear buffers now that we’ve copied data
        self.buf1 = None
        self.buf2 = None

        if not candidates:
            # Still publish current confirmed tracks and age/miss them
            self._predict_all(self.get_clock().now())
            self._maintain_tracks()
            self._publish_and_log([])
            return

        t_meas = max(stamp_candidates) if stamp_candidates else self.get_clock().now()

        # Predict tracks to measurement time
        self._predict_all(t_meas)

        # Associate with greedy nearest Mahalanobis gating
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_cands = list(range(len(candidates)))
        pairs = []
        if self.tracks:
            # Build cost as Mahalanobis distance^2
            costs = []
            for ti, tr in enumerate(self.tracks):
                y_list = []
                for ci, (z, _r, _d) in enumerate(candidates):
                    y, S, _ = tr.innovation(z, self.R)
                    try:
                        d2 = float(y.T @ np.linalg.inv(S) @ y)
                        y_list.append(d2)
                    except np.linalg.LinAlgError:
                        y_list.append(np.inf)
                costs.append(y_list)

            # Greedy: repeatedly pick lowest d2 under gate
            while True:
                best = (None, None, np.inf)
                for ti in unmatched_tracks:
                    for ci in unmatched_cands:
                        d2 = costs[ti][ci]
                        if d2 < best[2]:
                            best = (ti, ci, d2)
                ti, ci, d2 = best
                if ti is None or d2 > self.gate_chi2:
                    break
                pairs.append((ti, ci))
                unmatched_tracks.remove(ti)
                unmatched_cands.remove(ci)

        # Update matched and radius EMA
        for (ti, ci) in pairs:
            z, r_meas, _d = candidates[ci]
            self.tracks[ti].update(z, self.R)
            if np.isfinite(r_meas):
                if np.isfinite(self.tracks[ti].radius):
                    a = self.radius_alpha
                    self.tracks[ti].radius = (1.0 - a) * self.tracks[ti].radius + a * r_meas
                else:
                    self.tracks[ti].radius = r_meas
            if not self.tracks[ti].confirmed and self.tracks[ti].hits >= self.confirm_hits:
                self.tracks[ti].confirmed = True

        # For remaining unmatched candidates, attempt multi-sensor fuse or Euclidean fallback update
        still_unmatched = []
        for ci in unmatched_cands:
            z, r_meas, _d = candidates[ci]
            if not self.tracks:
                still_unmatched.append((z, r_meas))
                continue
            # nearest track in Euclidean sense
            dists = [np.linalg.norm(tr.x[0:3,0] - z) for tr in self.tracks]
            ti = int(np.argmin(dists))
            d_min = dists[ti]
            if d_min <= self.fuse_distance:
                # Treat as another measurement to the same track (sequential update)
                self.tracks[ti].update(z, self.R)
                if np.isfinite(r_meas):
                    if np.isfinite(self.tracks[ti].radius):
                        a = self.radius_alpha
                        self.tracks[ti].radius = (1.0 - a) * self.tracks[ti].radius + a * r_meas
                    else:
                        self.tracks[ti].radius = r_meas
                if not self.tracks[ti].confirmed and self.tracks[ti].hits >= self.confirm_hits:
                    self.tracks[ti].confirmed = True
            elif d_min <= self.merge_distance:
                # Conservative update with inflated noise
                R_inflated = self.R * 4.0
                self.tracks[ti].update(z, R_inflated)
                if np.isfinite(r_meas):
                    if np.isfinite(self.tracks[ti].radius):
                        a = self.radius_alpha
                        self.tracks[ti].radius = (1.0 - a) * self.tracks[ti].radius + a * r_meas
                    else:
                        self.tracks[ti].radius = r_meas
                if not self.tracks[ti].confirmed and self.tracks[ti].hits >= self.confirm_hits:
                    self.tracks[ti].confirmed = True
            else:
                still_unmatched.append((z, r_meas))

        # Create new tracks for remaining unmatched candidates
        for (z, r_meas) in still_unmatched:
            x0 = np.zeros((6,1))
            x0[0:3,0] = z
            P0 = np.diag([0.05,0.05,0.05, 0.1,0.1,0.1])
            self.tracks.append(Track(x0, P0, t_meas, self._next_id, r_meas))
            self._next_id += 1

        # Age unmatched tracks
        for ti in unmatched_tracks:
            self.tracks[ti].misses += 1

        # Merge close confirmed tracks
        self._merge_tracks_spatial()
        self._fuse_duplicate_tracks()

        # Maintain lifecycle
        self._maintain_tracks()

        # Publish + log
        self._publish_and_log(candidates)

    def _predict_all(self, t: Time):
        for tr in self.tracks:
            tr.predict(t, self.q_pos2, self.q_vel2)

    def _merge_tracks_spatial(self):
        keep = []
        for tr in sorted(self.tracks, key=lambda t: (t.confirmed, -t.hits), reverse=True):
            accept = True
            p = tr.x[0:3,0]
            for kept in keep:
                if np.linalg.norm(kept.x[0:3,0] - p) < self.merge_distance:
                    # keep the one with higher hits/confirmed
                    if (kept.confirmed and not tr.confirmed) or (kept.hits >= tr.hits):
                        accept = False
                        break
            if accept:
                keep.append(tr)
        self.tracks = keep

    def _fuse_duplicate_tracks(self):
        if not self.tracks:
            return
        # Only consider confirmed tracks for fusing
        tracks = [t for t in self.tracks if t.confirmed]
        others = [t for t in self.tracks if not t.confirmed]
        fused = []
        used = set()
        # Sort by hits descending to prefer long-lived IDs
        order = sorted(range(len(tracks)), key=lambda i: tracks[i].hits, reverse=True)
        for i in order:
            if i in used:
                continue
            base = tracks[i]
            group = [i]
            p0 = base.x[0:3,0]
            for j in order:
                if j == i or j in used:
                    continue
                p1 = tracks[j].x[0:3,0]
                if np.linalg.norm(p1 - p0) <= self.track_merge_m:
                    group.append(j)
            # Fuse group into base
            if len(group) > 1:
                # Weighted average by hits
                positions = []
                weights = []
                radii = []
                r_weights = []
                total_hits = 0
                min_misses = base.misses
                for idx in group:
                    t = tracks[idx]
                    positions.append(t.x[0:3,0])
                    weights.append(max(1, t.hits))
                    total_hits += t.hits
                    min_misses = min(min_misses, t.misses)
                    if np.isfinite(t.radius):
                        radii.append(t.radius)
                        r_weights.append(max(1, t.hits))
                W = np.array(weights, dtype=np.float64).reshape(-1,1)
                P = np.vstack([p.reshape(1,3) for p in positions])
                p_new = (W.T @ P) / np.sum(W)
                base.x[0:3,0] = p_new.reshape(3)
                base.hits = total_hits
                base.misses = min_misses
                if radii:
                    rw = np.array(r_weights, dtype=np.float64)
                    base.radius = float(np.dot(rw, np.array(radii, dtype=np.float64)) / np.sum(rw))
            fused.append(base)
            used.update(group)
        # Add back unconfirmed others
        fused.extend(others)
        self.tracks = fused

    def _maintain_tracks(self):
        alive = []
        for tr in self.tracks:
            if tr.misses <= self.max_misses:
                alive.append(tr)
        self.tracks = alive

    def _publish_and_log(self, candidates):
        out = Detection3DArray()
        out.header = Header()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.global_frame

        rows = []
        for tr in self.tracks:
            if not tr.confirmed:
                continue
            x,y,z = tr.x[0,0], tr.x[1,0], tr.x[2,0]
            det = Detection3D()
            det.header = out.header
            det.results.append(detections3d_hyp(x,y,z))
            r_for_bbox = tr.radius if np.isfinite(tr.radius) else self.default_radius
            det.bbox.size.x = r_for_bbox*2.0
            det.bbox.size.y = r_for_bbox*2.0
            det.bbox.size.z = r_for_bbox*2.0
            det.id = str(tr.id)
            out.detections.append(det)

            rows.append((out.header.stamp.sec, out.header.stamp.nanosec, x,y,z, (tr.radius if np.isfinite(tr.radius) else self.default_radius), tr.hits, 'global'))

        self.pub.publish(out)

        if rows:
            with open(self.csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                for row in rows:
                    w.writerow(row)

def detections3d_hyp(x,y,z):
    from vision_msgs.msg import ObjectHypothesisWithPose
    from geometry_msgs.msg import Pose, Quaternion
    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = 'circle'
    hyp.hypothesis.score = 1.0
    pose = Pose()
    pose.position.x = float(x); pose.position.y = float(y); pose.position.z = float(z)
    pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    hyp.pose.pose = pose
    return hyp

def main():
    rclpy.init()
    node = DedupeAndLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
