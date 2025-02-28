import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
import random


class SterlingLocalCostmap(Node):
    def __init__(self):
        super().__init__("sterling_local_costmap")
        self.occupany_grid_msg = None
        self.odometry_msg = None

        # Publishers
        self.sterling_costmap_publisher = self.create_publisher(OccupancyGrid, "sterling_local_costmap", 10)

        # Subscribers
        self.costmap_subscriber = self.create_subscription(OccupancyGrid, "/local_costmap/costmap", self.costmap_callback, 10)
        self.odometry_subscriber = self.create_subscription(Odometry, "/odometry/filtered", self.odometry_callback, 10)

        self.timer = self.create_timer(1.0, self.update_costmap)

    def costmap_callback(self, msg):
        self.occupany_grid_msg = msg
        
    def odometry_callback(self, msg):
        self.odometry_msg = msg

    def update_costmap(self):
        if not self.occupany_grid_msg:
            return

        msg = self.occupany_grid_msg

        # Modify msg.data to have random values between 0 and 100
        # msg.data = [0 for _ in range(int(len(msg.data) / 3))]

        self.sterling_costmap_publisher.publish(msg)
        self.get_logger().info(f"Length of data sent: {len(msg.data)}")


def main(args=None):
    rclpy.init(args=args)
    costmap_updater = SterlingLocalCostmap()
    rclpy.spin(costmap_updater)
    costmap_updater.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
