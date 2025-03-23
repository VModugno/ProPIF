from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch parameters
    gui_param = LaunchConfiguration('gui_enabled', default='true')
    camera_param = LaunchConfiguration('camera_enabled', default='true')
    sim_rate_param = LaunchConfiguration('simulation_rate', default='100.0')
    
    # Create simulation node
    simulation_node = Node(
        package='propif_simulation',
        executable='simulation_node',
        name='simulation_node',
        parameters=[{
            'gui_enabled': gui_param,
            'camera_enabled': camera_param,
            'simulation_rate': sim_rate_param,
            'robot_config': 'pandaconfig.json',
            'flower_model': 'flower.obj',
        }],
        output='screen'
    )
    
    # Return launch description
    return LaunchDescription([
        DeclareLaunchArgument('gui_enabled', default_value='true',
                              description='Enable PyBullet GUI'),
        DeclareLaunchArgument('camera_enabled', default_value='true',
                              description='Enable virtual camera'),
        DeclareLaunchArgument('simulation_rate', default_value='100.0',
                              description='Simulation update rate (Hz)'),
                              
        simulation_node
    ])