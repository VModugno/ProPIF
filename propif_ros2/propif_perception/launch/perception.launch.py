from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'use_sfm_reconstruction',
            default_value='false',
            description='Use SFM reconstruction (Plan B) instead of direct pose (Plan A)'
        ),
        
        DeclareLaunchArgument(
            'reconstruction_image_count',
            default_value='10',
            description='Number of images to collect for SFM reconstruction'
        ),
        
        DeclareLaunchArgument(
            'debug_windows',
            default_value='false',
            description='Show debug windows'
        ),

        # Perception node
        Node(
            package='propif_perception',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[{
                'classes': ['flower', 'leaf', 'tree', 'plant'],
                'use_sfm_reconstruction': LaunchConfiguration('use_sfm_reconstruction'),
                'reconstruction_image_count': LaunchConfiguration('reconstruction_image_count'),
                'debug_windows': LaunchConfiguration('debug_windows'),
            }]
        )
    ])
