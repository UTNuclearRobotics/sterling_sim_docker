from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sterling',
            executable='global_costmap_builder',
            name='global_costmap_builder',
            parameters=['config/config.yaml']
        )
    ])