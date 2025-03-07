import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node


class GlobalCostmapBuilder(Node):
    def __init__(self):
        super().__init__("global_costmap_builder")

        # Declare and get parameters
        self.declare_parameter("local_costmap_topic", "/local_costmap/costmap")
        self.declare_parameter("global_costmap_topic", "/global_costmap/costmap")

        self.local_costmap_topic = self.get_parameter("local_costmap_topic").value
        self.global_costmap_topic = self.get_parameter("global_costmap_topic").value

        # Subscribe to the local costmap
        self.create_subscription(
            OccupancyGrid,
            self.local_costmap_topic,
            self.local_costmap_callback,
            10,
        )

        # Subscribe to the global costmap
        self.create_subscription(
            OccupancyGrid,
            self.global_costmap_topic,
            self.global_costmap_callback,
            10,
        )

        # Publisher for the global costmap
        self.stitched_costmap_publisher = self.create_publisher(
            OccupancyGrid,
            "global_costmap",
            10,
        )

        # Initialize stitched costmap data
        self.stitched_costmap = None
        self.stitched_resolution = 0.0
        self.stitched_width = 0
        self.stitched_height = 0
        self.stitched_origin_x = 0.0
        self.stitched_origin_y = 0.0

    def global_costmap_callback(self, msg):
        """Callback for the global costmap."""
        self.stitched_costmap = msg
        self.stitched_resolution = msg.info.resolution
        self.stitched_width = msg.info.width
        self.stitched_height = msg.info.height
        self.stitched_origin_x = msg.info.origin.position.x
        self.stitched_origin_y = msg.info.origin.position.y

    def local_costmap_callback(self, msg):
        """Callback for the local costmap."""
        if self.stitched_costmap is None:
            self.get_logger().warn("Stitched costmap not yet initialized.")
            return

        # Extract local costmap data
        local_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        local_resolution = msg.info.resolution
        local_origin_x = msg.info.origin.position.x
        local_origin_y = msg.info.origin.position.y

        # Calculate the bounds of the local costmap in the global frame
        local_min_x = local_origin_x
        local_max_x = local_origin_x + msg.info.width * local_resolution
        local_min_y = local_origin_y
        local_max_y = local_origin_y + msg.info.height * local_resolution

        # Check if the local costmap exceeds the current stitched costmap bounds
        if (
            local_min_x < self.stitched_origin_x
            or local_max_x > self.stitched_origin_x + self.stitched_width * self.stitched_resolution
            or local_min_y < self.stitched_origin_y
            or local_max_y > self.stitched_origin_y + self.stitched_height * self.stitched_resolution
        ):
            # Resize the stitched costmap
            self.resize_stitched_costmap(local_min_x, local_max_x, local_min_y, local_max_y)

        # Transform local costmap data to stitched costmap frame
        for y in range(msg.info.height):
            for x in range(msg.info.width):
                # Calculate stitched coordinates
                stitched_x = int(
                    (x * local_resolution + local_origin_x - self.stitched_origin_x) / self.stitched_resolution
                )
                stitched_y = int(
                    (y * local_resolution + local_origin_y - self.stitched_origin_y) / self.stitched_resolution
                )

                # Ensure stitched coordinates are within bounds
                if 0 <= stitched_x < self.stitched_width and 0 <= stitched_y < self.stitched_height:
                    # Update stitched costmap (e.g., take the maximum value)
                    current_value = self.stitched_costmap.data[stitched_y * self.stitched_width + stitched_x]
                    new_value = local_data[y, x]
                    self.stitched_costmap.data[stitched_y * self.stitched_width + stitched_x] = max(
                        current_value, new_value
                    )

        # Publish the updated global costmap
        self.stitched_costmap_publisher.publish(self.stitched_costmap)

    def resize_stitched_costmap(self, min_x, max_x, min_y, max_y):
        """Resize the stitched costmap to accommodate new data."""
        # Calculate new stitched costmap size and origin
        new_origin_x = min(min_x, self.stitched_origin_x)
        new_origin_y = min(min_y, self.stitched_origin_y)
        new_width = int((max_x - new_origin_x) / self.stitched_resolution) + 1
        new_height = int((max_y - new_origin_y) / self.stitched_resolution) + 1

        # Create a new stitched costmap grid
        new_stitched_data = np.zeros((new_height, new_width), dtype=np.int8)

        # Calculate the offset for the existing data
        offset_x = int((self.stitched_origin_x - new_origin_x) / self.stitched_resolution)
        offset_y = int((self.stitched_origin_y - new_origin_y) / self.stitched_resolution)

        # Copy the existing data to the new grid
        for y in range(self.stitched_height):
            for x in range(self.stitched_width):
                new_x = x + offset_x
                new_y = y + offset_y
                if 0 <= new_x < new_width and 0 <= new_y < new_height:
                    new_stitched_data[new_y, new_x] = self.stitched_costmap.data[y * self.stitched_width + x]

        # Update the stitched costmap properties
        self.stitched_origin_x = new_origin_x
        self.stitched_origin_y = new_origin_y
        self.stitched_width = new_width
        self.stitched_height = new_height

        # Update the stitched costmap message
        self.stitched_costmap.info.width = new_width
        self.stitched_costmap.info.height = new_height
        self.stitched_costmap.info.origin.position.x = new_origin_x
        self.stitched_costmap.info.origin.position.y = new_origin_y
        self.stitched_costmap.data = new_stitched_data.flatten().tolist()

        self.get_logger().info(f"Resized stitched costmap to {new_width}x{new_height}")


def main(args=None):
    rclpy.init(args=args)
    node = GlobalCostmapBuilder()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
