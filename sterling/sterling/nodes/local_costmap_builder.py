import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

from sterling.bev import get_BEV_image
from sterling.bev_costmap import BEVCostmap


class LocalCostmapBuilder(Node):
    def __init__(self):
        super().__init__("local_costmap_builder")

        # Declare parameters with default values
        self.declare_parameter("camera_topic", "/oakd2/oak_d_node/rgb/image_rect_color")
        self.declare_parameter("local_costmap_topic", "/local_costmap/costmap")
        self.declare_parameter("model_path", "path/to/terrain_representation_model.pt")
        self.declare_parameter("homography_matrix", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.declare_parameter("patch_size_px", 128)
        self.declare_parameter("patch_size_m", 0.23)
        self.declare_parameter("base_link_offset_m", 1.4)

        # Get parameter values
        self.camera_topic = self.get_parameter("camera_topic").value
        self.local_costmap_topic = self.get_parameter("local_costmap_topic").value
        model_path = self.get_parameter("model_path").value
        self.H = np.array(self.get_parameter("homography_matrix").value).reshape(3, 3)
        self.patch_size_px = self.get_parameter("patch_size_px").value
        self.patch_size_m = self.get_parameter("patch_size_m").value
        self.base_link_offset_m = self.get_parameter("base_link_offset_m").value

        # Print parameter values
        self.get_logger().debug(f"Camera topic: {self.camera_topic}")
        self.get_logger().debug(f"Local costmap topic: {self.local_costmap_topic}")
        self.get_logger().debug(f"Model path: {model_path}")
        self.get_logger().debug(f"Homography matrix: \n{self.H}")
        self.get_logger().debug(f"Patch size (px): {self.patch_size_px}")
        self.get_logger().debug(f"Patch size (m): {self.patch_size_m}")
        self.get_logger().debug(f"Base link offset (m): {self.base_link_offset_m}")

        # Subscribers
        self.camera_subscriber = self.create_subscription(Image, self.camera_topic, self.camera_callback, 10)
        self.costmap_subscriber = self.create_subscription(
            OccupancyGrid, self.local_costmap_topic, self.costmap_callback, 10
        )

        # Publishers
        self.sterling_costmap_publisher = self.create_publisher(OccupancyGrid, "local_costmap", 10)

        # Timers
        self.timer = self.create_timer(1.0, self.update_costmap)

        # Initialize tf buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_terrain_preferred_costmap = BEVCostmap(model_path).BEV_to_costmap

        self.LocalCostmapHelper = None

        # Buffers to store and fetch latest message
        self.camera_msg = None
        self.yaw_angle = None
        self.occupany_grid_msg = None

    def camera_callback(self, msg):
        self.camera_msg = msg

        # Lookup transform from base_link to get orientation
        try:
            transform = self.tf_buffer.lookup_transform("base_link", "map", rclpy.time.Time())
            self.yaw_angle = LocalCostmapHelper.quarternion_to_euler(transform.transform.rotation)
            # self.get_logger().info(f"Yaw angle: {np.degrees(yaw_angle)}")
        except Exception as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
            return

    def costmap_callback(self, msg):
        if self.LocalCostmapHelper is None:
            self.LocalCostmapHelper = LocalCostmapHelper(msg.info.resolution, msg.info.width, msg.info.height)
        self.occupany_grid_msg = msg

    def update_costmap(self):
        """
        Use the rolling window of the local costmap to stitch the terrain preferred local costmap.
        """
        if not self.camera_msg or not self.yaw_angle or not self.occupany_grid_msg:
            if self.camera_msg is None:
                self.get_logger().debug("Camera message is None")
            if self.yaw_angle is None:
                self.get_logger().debug("Yaw angle is None")
            if self.occupany_grid_msg is None:
                self.get_logger().debug("Occupancy grid message is None")
            self.get_logger().info("Waiting for camera and occupancy grid message...")
            return

        # Get BEV image
        image_data = np.frombuffer(self.camera_msg.data, dtype=np.uint8).reshape(
            self.camera_msg.height, self.camera_msg.width, -1
        )
        # Preview the image using OpenCV
        bev_image = get_BEV_image(image_data, self.H, (self.patch_size_px, self.patch_size_px), (7, 12))

        # Get terrain preferred costmap
        terrain_costmap = self.get_terrain_preferred_costmap(bev_image, self.patch_size_px)
        # self.get_logger().info(f"Costmap:\n{terrain_costmap}")

        # TODO: Bug that the costmap is flipped horizontally
        terrain_costmap = np.fliplr(terrain_costmap)

        # Set costs in the region
        data_2d = self.LocalCostmapHelper.set_costs_in_region(
            0, -self.base_link_offset_m, self.patch_size_m, terrain_costmap
        )

        # Rotate the costmap by the yaw angle
        rotated_data = LocalCostmapHelper.rotate_costmap(data_2d, np.degrees(self.yaw_angle) - 90)
        rotated_data = np.array(rotated_data).flatten()

        # Keep the highest cost when stitching the local costmap
        msg = self.occupany_grid_msg
        msg.data = rotated_data.tolist()
        # msg.data = np.maximum(msg.data, rotated_data).tolist()

        # Publish message
        self.sterling_costmap_publisher.publish(msg)

        # Reset the buffers
        # self.camera_msg = None
        # self.yaw_angle = None
        # self.occupany_grid_msg = None


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
        canvas = np.full((self.height_cells, self.width_cells), -1, dtype=int)

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
        height, width = data_2d.shape
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
    rclpy.init(args=args)
    costmap_updater = LocalCostmapBuilder()
    rclpy.spin(costmap_updater)
    costmap_updater.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
