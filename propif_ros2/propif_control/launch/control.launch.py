from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():    
    # Create control node
    control_node = Node(
        package='propif_control',
        executable='control_node',
        name='control_node',
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