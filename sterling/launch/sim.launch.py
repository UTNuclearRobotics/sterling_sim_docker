from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config_file = os.path.join(get_package_share_directory("sterling"), "config", "config.yaml")

    return LaunchDescription(
        [
            Node(
                package="sterling",
                executable="global_costmap_builder",
                name="global_costmap_builder",
                parameters=[config_file],
            )
        ]
    )
