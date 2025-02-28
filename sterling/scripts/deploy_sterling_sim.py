import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
import yaml
from bev_costmap import BEVCostmap
from bev import get_BEV_image
import os


class DeploySterlingSim(Node):
    def __init__(self, config):
        super().__init__("deploy_sterling_sim")
        HOMOGRAPHY = config["homography"]
        DEPLOY_NODE = config["deploy_sterling_sim"]

        # Topics
        self.camera_topic = DEPLOY_NODE["camera_topic"]
        self.odometry_topic = DEPLOY_NODE["odometry_topic"]
        self.local_costmap_topic = DEPLOY_NODE["local_costmap_topic"]

        # Subscribers
        self.camera_subscriber = self.create_subscription(Image, self.camera_topic, self.camera_callback, 10)
        self.odometry_subscriber = self.create_subscription(Odometry, self.odometry_topic, self.odometry_callback, 10)
        self.costmap_subscriber = self.create_subscription(
            OccupancyGrid, self.local_costmap_topic, self.costmap_callback, 10
        )

        # Publishers
        self.sterling_costmap_publisher = self.create_publisher(OccupancyGrid, "sterling_local_costmap", 10)

        # Timers
        self.timer = self.create_timer(1.0, self.update_costmap)

        H = np.array(HOMOGRAPHY["matrix"])
        self.H_INV = np.linalg.inv(H)

        self.get_terrain_preferred_costmap = BEVCostmap(
            DEPLOY_NODE["terrain_representation_model"], DEPLOY_NODE["kmeans_model"], DEPLOY_NODE["terrain_preferences"]
        ).BEV_to_costmap

        self.camera_msg = None
        self.odometry_msg = None
        self.occupany_grid_msg = None

    def camera_callback(self, msg):
        self.camera_msg = msg

    def odometry_callback(self, msg):
        self.odometry_msg = msg

    def costmap_callback(self, msg):
        self.occupany_grid_msg = msg

    def update_costmap(self):
        if not self.camera_msg or not self.odometry_msg or not self.occupany_grid_msg:
            return

        # Get BEV image
        image_data = np.frombuffer(self.camera_msg.data, dtype=np.uint8).reshape(self.camera_msg.height, self.camera_msg.width, -1)
        bev_image = get_BEV_image(image_data, self.H_INV, (128, 128), (7, 12))

        # Get terrain preferred costmap
        terrain_preferred_costmap = self.get_terrain_preferred_costmap(bev_image, 128)
        self.get_logger().info(f"Costmap:\n{terrain_preferred_costmap}")
        
        msg = self.occupany_grid_msg
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        # Create an empty 2d numpy array
        data_2d = np.zeros((width, height), dtype=int)
        
        

        # Rotate the costmap by the yaw angle
        yaw_angle = DeploySterlingSim.quarternion_to_euler(self.odometry_msg.pose.pose.orientation)
        # self.get_logger().info(f"Yaw angle: {np.degrees(yaw_angle)}")
        msg.data = self.rotate_costmap(data_2d, -np.degrees(yaw_angle) + 90)

        # Publish message
        self.sterling_costmap_publisher.publish(msg)

    @staticmethod
    def quarternion_to_euler(orientation_q):
        """
        Convert quaternion to Euler angles
        Args:
            orientation_q: Quaternion object
        Returns:
            Yaw angle in radians
        """
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        yaw_angle = np.arctan2(siny_cosp, cosy_cosp)

        return yaw_angle

    @staticmethod
    def rotate_costmap(data_2d, angle):
        """
        Rotate a costmap by a given angle
        Args:
            data_2d: 2D numpy array
            angle: Angle in degrees
        Returns:
            rotated_data: 1D list
        """
        width, height = data_2d.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_data = cv2.warpAffine(
            data_2d,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1,
        )

        return rotated_data.flatten().tolist()


def main(args=None):
    # Load the config
    script_dir = os.path.dirname(__file__)
    config_file = os.path.join(script_dir, "../config/config.yaml")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    rclpy.init(args=args)
    costmap_updater = DeploySterlingSim(config)
    rclpy.spin(costmap_updater)
    costmap_updater.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
