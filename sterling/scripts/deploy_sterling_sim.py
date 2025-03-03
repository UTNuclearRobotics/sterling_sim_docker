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

        self.H = np.array(HOMOGRAPHY["matrix"])

        self.get_terrain_preferred_costmap = BEVCostmap(
            DEPLOY_NODE["terrain_representation_model"], DEPLOY_NODE["kmeans_model"], DEPLOY_NODE["terrain_preferences"]
        ).BEV_to_costmap

        self.LocalCostmapHelper = None

        # Buffers to store and fetch latest message
        self.camera_msg = None
        self.odometry_msg = None
        self.occupany_grid_msg = None
        
        self.patch_size_px = (128, 128)
        self.patch_size_m = (0.23, 0.23)
        self.base_link_offset_m = 1.4

    def camera_callback(self, msg):
        self.camera_msg = msg

    def odometry_callback(self, msg):
        self.odometry_msg = msg

    def costmap_callback(self, msg):
        if self.LocalCostmapHelper is None:
            self.LocalCostmapHelper = LocalCostmapHelper(msg.info.resolution, msg.info.width, msg.info.height)
        self.occupany_grid_msg = msg

    def update_costmap(self):
        if not self.camera_msg or not self.odometry_msg or not self.occupany_grid_msg:
            return

        # Get BEV image
        image_data = np.frombuffer(self.camera_msg.data, dtype=np.uint8).reshape(
            self.camera_msg.height, self.camera_msg.width, -1
        )
        # Preview the image using OpenCV
        bev_image = get_BEV_image(image_data, self.H, self.patch_size_px, (7, 12))

        # Get terrain preferred costmap
        terrain_costmap = self.get_terrain_preferred_costmap(bev_image, self.patch_size_px[0])
        self.get_logger().debug(f"Costmap:\n{terrain_costmap}")

        # Set costs in the region
        data_2d = self.LocalCostmapHelper.set_costs_in_region(0, -self.base_link_offset_m, self.patch_size_m[0], terrain_costmap)

        # Rotate the costmap by the yaw angle
        yaw_angle = LocalCostmapHelper.quarternion_to_euler(self.odometry_msg.pose.pose.orientation)
        # self.get_logger().info(f"Yaw angle: {np.degrees(yaw_angle)}")
        rotated_data = LocalCostmapHelper.rotate_costmap(data_2d, -np.degrees(yaw_angle) - 90)
        
        msg = self.occupany_grid_msg
        msg.data = [int(val) for val in np.array(rotated_data).flatten()]

        # Publish message
        self.sterling_costmap_publisher.publish(msg)


class LocalCostmapHelper:
    def __init__(self, resolution=0.05, width_cells=120, height_cells=120):
        # Local costmap dimensions and resolution
        self.resolution = resolution  # 5 cm per cell
        self.width_cells = width_cells
        self.height_cells = height_cells
        self.width_m = width_cells * resolution  # 6 meters
        self.height_m = height_cells * resolution  # 6 meters

        # Center of the local costmap in cell coordinates
        self.center_x = self.width_cells // 2  # 60 cells
        self.center_y = self.height_cells // 2  # 60 cells

    def set_costs_in_region(self, x_m, y_m, cell_size_m, terrain_costmap):
        upscale_factor = int(cell_size_m / self.resolution)

        # Convert meters to cells
        offset_x_cells = int(x_m / self.resolution)
        offset_y_cells = int(y_m / self.resolution)
        width_cells = upscale_factor * len(terrain_costmap[0])
        height_cells = upscale_factor * len(terrain_costmap)

        # Calculate the bottom-left corner of the region in cell coordinates
        # x = self.center_x + offset_x_cells
        x = self.center_x + offset_x_cells - width_cells // 2
        # y = self.center_y + offset_y_cells
        y = self.center_y + offset_y_cells - height_cells

        # Scale the data array to account for the resolution
        return self.upsample_2d_array(terrain_costmap, upscale_factor, x, y)

    def upsample_2d_array(self, arr, factor, x_start, y_start):
        """
        Upsample a 2D array by a factor.

        Args:
            arr (list of list): The original 2D array.

        Returns:
            list of list: The upsampled 2D array.
        """
        canvas = np.zeros((self.width_cells, self.height_cells), dtype=int)

        # Get the dimensions of the original array
        height = len(arr)
        width = len(arr[0]) if height > 0 else 0

        # Fill the upsampled array
        for i in range(height):
            for j in range(width):
                # Get the value from the original array
                value = arr[i][j]

                # Fill the corresponding block in the upsampled array
                for di in range(factor):
                    for dj in range(factor):
                        canvas[y_start + factor * i + di][x_start + factor * j + dj] = value

        return canvas

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

        return rotated_data


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
