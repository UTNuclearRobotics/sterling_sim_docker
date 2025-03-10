import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node


class GlobalCostmapBuilder(Node):
    """
    Listens to the global costmap know origin and when to resize the map.
    The only values being placed onto the global costamp are stitched sterling local costmaps.
    
    Requires local_costmap, global_costmap, and slam_toolbox map resolutions to be the same.
    """

    def __init__(self):
        super().__init__("global_costmap_builder")

        # Declare and get parameters
        self.declare_parameter("local_costmap_topic", "/local_costmap/costmap")
        self.declare_parameter("global_costmap_topic", "/global_costmap/costmap")

        self.local_costmap_topic = self.get_parameter("local_costmap_topic").value
        self.global_costmap_topic = self.get_parameter("global_costmap_topic").value

        # Print parameter values
        self.get_logger().info(f"Local costmap topic: {self.local_costmap_topic}")
        self.get_logger().info(f"Global costmap topic: {self.global_costmap_topic}")

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

        # Initialize stitched costmap
        self.stitched_costmap = None

    def global_costmap_callback(self, msg):
        """Callback for the global costmap."""
        self.global_msg = msg
        self.global_resolution = msg.info.resolution
        self.global_width = msg.info.width
        self.global_height = msg.info.height
        self.global_origin_x = msg.info.origin.position.x
        self.global_origin_y = msg.info.origin.position.y

        if self.stitched_costmap is None:
            self.stitched_costmap = np.full((self.global_height, self.global_width), -1, dtype=int)
            self.stitched_resolution = self.global_resolution
            self.stitched_width = self.global_width
            self.stitched_height = self.global_height
            self.stitched_origin_x = self.global_origin_x
            self.stitched_origin_y = self.global_origin_y
        elif self.stitched_width != self.global_width or self.stitched_height != self.global_height:
            self.resize_stitched_costmap()

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
                    # Update stitched costmap
                    current_value = self.stitched_costmap[stitched_y, stitched_x]
                    new_value = local_data[y, x]
                    self.stitched_costmap[stitched_y, stitched_x] = max(current_value, new_value)

        # Publish the updated global costmap
        update_msg = self.global_msg
        update_msg.data = self.stitched_costmap.flatten().tolist()
        self.stitched_costmap_publisher.publish(update_msg)

    def resize_stitched_costmap(self):
        """Resize the stitched costmap to accommodate new data."""

        # Create a new stitched costmap grid
        new_stitched_costmap = np.full((self.global_height, self.global_width), -1, dtype=np.int8)

        # Calculate the offset for the existing data
        offset_x = int((self.stitched_origin_x - self.global_origin_x) / self.stitched_resolution)
        offset_y = int((self.stitched_origin_y - self.global_origin_y) / self.stitched_resolution)

        # Copy the existing data to the new grid
        for y in range(self.stitched_height):
            for x in range(self.stitched_width):
                new_x = x + offset_x
                new_y = y + offset_y
                if 0 <= new_x < self.global_width and 0 <= new_y < self.global_height:
                    new_stitched_costmap[new_y, new_x] = self.stitched_costmap[y, x]

        # Update the stitched costmap properties
        self.stitched_width = self.global_width
        self.stitched_height = self.global_height
        
        # Update the stitched costmap
        self.stitched_costmap = new_stitched_costmap
        self.get_logger().info(f"Resized stitched costmap to {self.stitched_width}x{self.stitched_height}")


def main(args=None):
    rclpy.init(args=args)
    node = GlobalCostmapBuilder()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
