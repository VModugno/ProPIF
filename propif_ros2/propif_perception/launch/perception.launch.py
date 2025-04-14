from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='propif_perception',
            executable='perception_node',
            name='perception_node',
            output='screen'
        )
    ])
