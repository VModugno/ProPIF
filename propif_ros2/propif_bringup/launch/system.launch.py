from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Define parameters
    use_sim = LaunchConfiguration('use_simulation', default='true')
    use_gui = LaunchConfiguration('gui_enabled', default='true')
    
    # Find package paths
    simulation_pkg = FindPackageShare('propif_simulation')
    perception_pkg = FindPackageShare('propif_perception')
    control_pkg = FindPackageShare('propif_control')
    
    # Create simulation launch description
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([simulation_pkg, 'launch', 'simulation.launch.py'])
        ]),
        launch_arguments={
            'gui_enabled': use_gui,
        }.items()
    )
    
    # Create perception launch description
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([perception_pkg, 'launch', 'perception.launch.py'])
        ])
    )
    
    # Create control launch description
    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([control_pkg, 'launch', 'control.launch.py'])
        ]),
        launch_arguments={
            'use_simulation': use_sim,
        }.items()
    )
    
    # Return complete launch description
    return LaunchDescription([
        # Declare parameters
        DeclareLaunchArgument('use_simulation', default_value='true',
                              description='Use simulation (true) or real hardware (false)'),
        DeclareLaunchArgument('gui_enabled', default_value='true',
                              description='Enable PyBullet GUI'),
                              
        # Launch components
        simulation_launch,
        # perception_launch,
        control_launch,
    ])