import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
import random
import numpy as np
import cv2


class LocalCostmapHelper:
    def __init__(self, resolution=0.05, width_m=6.0, height_m=6.0):
        # Local costmap dimensions and resolution
        self.resolution = resolution  # 5 cm per cell
        self.width_m = width_m  # 6 meters
        self.height_m = height_m  # 6 meters
        self.width_cells = int(self.width_m / self.resolution)  # 120 cells
        self.height_cells = int(self.height_m / self.resolution)  # 120 cells

        # Center of the local costmap in cell coordinates
        self.center_x = self.width_cells // 2  # 60 cells
        self.center_y = self.height_cells // 2  # 60 cells

        self.blank_costmap = np.zeros((int(width_m / resolution), int(height_m / resolution)), dtype=int)


class SterlingLocalCostmap(Node):
    def __init__(self):
        super().__init__("sterling_local_costmap")
        # Publishers
        self.sterling_costmap_publisher = self.create_publisher(OccupancyGrid, "sterling_local_costmap", 10)

        # Subscribers
        self.costmap_subscriber = self.create_subscription(
            OccupancyGrid, "/local_costmap/costmap", self.costmap_callback, 10
        )
        self.odometry_subscriber = self.create_subscription(Odometry, "/odometry/filtered", self.odometry_callback, 10)

        self.timer = self.create_timer(1.0, self.update_costmap)
        
        self.local_costmap_helper = LocalCostmapHelper()
        self.occupany_grid_msg = None
        self.odometry_msg = None

    def costmap_callback(self, msg):
        self.occupany_grid_msg = msg

    def odometry_callback(self, msg):
        self.odometry_msg = msg

    def update_costmap(self):
        if not self.occupany_grid_msg or not self.odometry_msg:
            return

        msg = self.occupany_grid_msg
        
        # Convert quaternion to Euler angles
        orientation_q = self.odometry_msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        yaw_angle = np.arctan2(siny_cosp, cosy_cosp)

        # Rotate the costmap by the yaw angle
        rotated_msg = self.rotate_costmap(msg, 45)

        # width = msg.info.width
        # height = msg.info.height
        # data = msg.data

        # # Convert the 1D data array into a 2D array
        # data_2d = [data[i * width:(i + 1) * width] for i in range(height)]

        # # Draw a triangle from the center to the top filled with 100's
        # center_x = width // 2
        # center_y = height // 2
        # for y in range(center_y):
        #     for x in range(center_x - y, center_x + y + 1):
        #         data_2d[y][x] = 100

        # # Flatten the 2D array back into a 1D array
        # msg.data = [item for sublist in data_2d for item in sublist]

        self.sterling_costmap_publisher.publish(rotated_msg)
        self.get_logger().info(f"Length of data sent: {len(msg.data)}")

    def rotate_costmap(self, msg, angle):
        # Convert the occupancy grid data to a numpy array
        width = msg.info.width
        height = msg.info.height
        data = np.zeros((width, height), dtype=int)

        # Rotate the numpy array using OpenCV
        center = (width // 2, height // 2)
        for y in range(center[1]):
            for x in range(center[0] - y, center[0] + y + 1):
                data[y][x] = 100
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_data = cv2.warpAffine(
            data,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1,
        )

        # Create a new OccupancyGrid message for the rotated costmap
        rotated_msg = OccupancyGrid()
        rotated_msg.header = msg.header
        rotated_msg.info = msg.info
        rotated_msg.data = rotated_data.flatten().tolist()

        return rotated_msg


def main(args=None):
    rclpy.init(args=args)
    costmap_updater = SterlingLocalCostmap()
    rclpy.spin(costmap_updater)
    costmap_updater.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
