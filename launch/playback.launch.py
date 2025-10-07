# launch/playback.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare the 'use_sim_time' argument.
    # This is set to 'true' by default for playback.
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (bag file) time'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Common parameters for both detector nodes
    common_params = [{'global_frame': 'map'},
                     {'publish_debug': True}, # Enable debug image for analysis
                     {'depth_scale': 1.0},
                     {'use_sim_time': use_sim_time}]

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
                    'use_msg_frame': True,
                    'dedupe_threshold_m': 0.1,
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
                    'use_msg_frame': True,
                    'dedupe_threshold_m': 0.1,
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
                               'log_dir':'circle_logs_playback'},
                              {'use_sim_time': use_sim_time}])
    
    markers = Node(
        package='md_circles',
        executable='circle_marker_node',
        name='circle_marker_node',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        use_sim_time_arg,
        det1, 
        det2, 
        dedupe,  
        markers
    ])
