# launch/circles.launch.py
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
import os, datetime
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_md_circles = get_package_share_directory('md_circles')

    # Common parameters for both detector nodes
    common_params = [{'global_frame': 'map'},
                     {'publish_debug': True},
                     {'depth_scale': 1.0}]  # set 0.001 if depth in mm

    # Other common arguments for the nodes
    common_node_args = {'output': 'screen'}

    det1 = Node(package='md_circles', executable='circle_detector_node',
                name='circle_detector_drone1',
                namespace='drone1',
                parameters=common_params + [{
                    'rgb_topic': '/drone1/rgb_camera',
                    'depth_topic': '/drone1/depth_camera',
                    'camera_info_topic': '/drone1/camera_info',
                    'camera_frame': 'drone1/camera_link',
                    'drone_id': 'drone1'}],
                **common_node_args)

    det2 = Node(package='md_circles', executable='circle_detector_node',
                name='circle_detector_drone2',
                namespace='drone2',
                parameters=common_params + [{
                    'rgb_topic': '/drone2/rgb_camera',
                    'depth_topic': '/drone2/depth_camera',
                    'camera_info_topic': '/drone2/camera_info',
                    'camera_frame': 'drone2/camera_link',
                    'drone_id': 'drone2'}],
                **common_node_args)

    dedupe = Node(package='md_circles', executable='dedupe_and_logger_node',
                  name='dedupe_and_logger', output='screen',
                  parameters=[{'global_frame':'map',
                               'gate_chi2': 30.0,
                               'R_diag_m': [0.08, 0.08, 0.08],
                               'merge_distance_m': 0.25,
                               'confirm_hits': 3,
                               'fuse_distance_m': 0.3,
                               'track_merge_m': 0.6,
                               'default_radius_m': 0.15,
                               'log_dir':'circle_logs'}])
    
    # Since no SLAM is running, we must publish static transforms to define
    # the position of each drone's camera in the map frame. This is for
    # testing. In a real scenario, a localization system (like SLAM)
    # would provide these transforms dynamically.
    # The arguments are: x y z yaw pitch roll parent_frame_id child_frame_id
    static_tf_drone1 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_pub_drone1',
        arguments=['0', '0', '0', '0', '0', '0', 'drone1/map', 'drone1/odom']
    )

    static_tf_drone2 = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_pub_drone2',
        arguments=['0', '0', '0', '0', '0', '0', 'drone2/map', 'drone2/odom']
    )

    # Node to publish visualization markers for RViz2
    markers = Node(
        package='md_circles',
        executable='circle_marker_node',
        name='circle_marker_node'
    )

    qos_profile_path = os.path.join(pkg_md_circles, 'config', 'qos_profiles.yaml')

    # Create a timestamped directory for the bag file
    bag_output_dir = os.path.join(
        '/home/frazergene/drone/drone_ws/src/md_circles/resource',
        'circles_bag/bag_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )

    # Action to record specified topics to a rosbag.
    # The -o argument specifies the output bag name.
    record_bag = ExecuteProcess(
        cmd=[
            'ros2','bag','record',
            '-s','mcap',
            '--compression-mode','file','--compression-format','zstd',
            '--max-bag-size','4096',            # MB per file
            '--max-bag-duration','900',         # seconds per file
            '--qos-profile-overrides-path', qos_profile_path,
            '-o', bag_output_dir,
            # drone1
            # '/drone1/odometry/in',
            '/drone1/rgb_camera',
            '/drone1/depth_camera',
            '/drone1/camera_info',
            # '/drone1/depth_camera/camera_info',   # <-- add if available
            # drone2
            # '/drone2/odometry/in',
            '/drone2/rgb_camera',
            '/drone2/depth_camera',
            '/drone2/camera_info',
            # '/drone2/depth_camera/camera_info',   # <-- add if available
            # transforms & time
            '/tf','/tf_static','/clock'
        ],
        output='screen'
    )

    delayed_record_bag = TimerAction(
        period=5.0,
        actions=[record_bag]
    )

    return LaunchDescription([
        # det1, det2, dedupe, markers,
        static_tf_drone1, static_tf_drone2,
        delayed_record_bag
    ])

if __name__ == '__main__':
    generate_launch_description()
