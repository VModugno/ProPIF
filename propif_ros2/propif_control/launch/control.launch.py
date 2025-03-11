from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch parameters
    use_sim_param = LaunchConfiguration('use_simulation', default='true')
    control_freq_param = LaunchConfiguration('control_frequency', default='100.0')
    approach_dist_param = LaunchConfiguration('approach_distance', default='0.1')
    interaction_time_param = LaunchConfiguration('interaction_duration', default='5.0')
    
    # Get package share directory
    pkg_dir = get_package_share_directory('propif_control')
    
    # Create control node
    control_node = Node(
        package='propif_control',
        executable='control_node',
        name='control_node',
        parameters=[{
            'use_simulation': use_sim_param,
            'control_frequency': control_freq_param,
            'approach_distance': approach_dist_param,
            'interaction_duration': interaction_time_param,
        }],
        output='screen'
    )
    
    # Return launch description
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument('use_simulation', default_value='true',
                            description='Use simulation (true) or real hardware (false)'),
        DeclareLaunchArgument('control_frequency', default_value='100.0',
                            description='Control loop frequency (Hz)'),
        DeclareLaunchArgument('approach_distance', default_value='0.1',
                            description='Approach distance to the plane (m)'),
        DeclareLaunchArgument('interaction_duration', default_value='5.0',
                            description='Duration of plant interaction (s)'),
                            
        # Nodes
        control_node
    ])