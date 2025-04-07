from launch import LaunchDescription
from launch_ros.actions import Node

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
        # Nodes
        control_node
    ])