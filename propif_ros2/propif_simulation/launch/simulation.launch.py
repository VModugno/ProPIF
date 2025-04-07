from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():    
    # Create simulation node
    simulation_node = Node(
        package='propif_simulation',
        executable='simulation_node',
        name='simulation_node',
        output='screen'
    )
    
    # Return launch description
    return LaunchDescription([                              
        simulation_node
    ])