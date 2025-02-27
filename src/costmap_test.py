import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
import random


class SterlingLocalCostmap(Node):
    def __init__(self):
        super().__init__("sterling_local_costmap")
        self.occupany_grid_msg = None

        # ROS publishers
        self.costmap_publisher = self.create_publisher(OccupancyGrid, "costmap_black", 10)

        # self.map_subscriber = self.create_subscription(OccupancyGrid, "/local_costmap/costmap", self.map_callback, 10)
        self.map_subscriber = self.create_subscription(OccupancyGrid, "/map", self.map_callback, 10)

        self.timer = self.create_timer(1.0, self.update_costmap)

    def map_callback(self, msg):
        self.occupany_grid_msg = msg

    def update_costmap(self):
        if not self.occupany_grid_msg:
            return

        msg = self.occupany_grid_msg

        # Modify msg.data to have random values between 0 and 100
        msg.data = [0 for _ in range(len(msg.data))]

        self.costmap_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    costmap_updater = SterlingLocalCostmap()
    rclpy.spin(costmap_updater)
    costmap_updater.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
