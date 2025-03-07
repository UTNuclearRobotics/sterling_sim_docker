import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node


class GlobalCostmapBuilder(Node):
    def __init__(self):
        super().__init__("global_costmap_builder")

        # Declare and get parameters
        self.local_costmap_topic = self.get_parameter("local_costmap_topic").value
        self.global_costmap_topic = self.get_parameter("global_costmap_topic").value

        # Subscribe to the local costmap
        self.subscription = self.create_subscription(
            OccupancyGrid,
            self.local_costmap_topic,
            self.local_costmap_callback,
            10,
        )
        
        # Subscribe to the global costmap
        self.subscription = self.create_subscription(
            OccupancyGrid,
            self.global_costmap_topic,
            self.global_costmap_callback,
            10,
        )

        # Publisher for the global costmap
        self.publisher = self.create_publisher(
            OccupancyGrid,
            "/global_costmap",
            10,
        )
        
        self.global_costmap_msg = None
        
    def global_costmap_callback(self, msg):
        self.global_costmap_msg = msg
        self.global_origin_x = msg.info.origin.position.x
        self.global_origin_y = msg.info.origin.position.y
        self.global_resolution = msg.info.resolution
        self.global_width = msg.info.width
        self.global_height = msg.info.height
        
    def local_costmap_callback(self, msg):
        # Extract local costmap data
        local_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)

        # Get local costmap origin in global frame
        local_origin_x = msg.info.origin.position.x
        local_origin_y = msg.info.origin.position.y

        # Calculate the bounds of the local costmap in the global frame
        local_min_x = local_origin_x
        local_max_x = local_origin_x + msg.info.width * msg.info.resolution
        local_min_y = local_origin_y
        local_max_y = local_origin_y + msg.info.height * msg.info.resolution

        # Check if the local costmap exceeds the current global costmap bounds
        if (
            local_min_x < self.global_origin_x
            or local_max_x > self.global_origin_x + self.global_width * self.global_resolution
            or local_min_y < self.global_origin_y
            or local_max_y > self.global_origin_y + self.global_height * self.global_resolution
        ):
            # Resize the global costmap
            self.resize_global_costmap(local_min_x, local_max_x, local_min_y, local_max_y)

        # Transform local costmap data to global frame
        for y in range(msg.info.height):
            for x in range(msg.info.width):
                # Calculate global coordinates
                global_x = int(
                    (x * msg.info.resolution + local_origin_x - self.global_origin_x) / self.global_resolution
                )
                global_y = int(
                    (y * msg.info.resolution + local_origin_y - self.global_origin_y) / self.global_resolution
                )

                # Ensure global coordinates are within bounds
                if 0 <= global_x < self.global_width and 0 <= global_y < self.global_height:
                    # Update global costmap (e.g., take the maximum value)
                    self.global_grid[global_y, global_x] = max(self.global_grid[global_y, global_x], local_data[y, x])

        # Publish the updated global costmap
        self.publish_global_costmap()

    def resize_global_costmap(self, min_x, max_x, min_y, max_y):
        # Calculate new global costmap size and origin
        new_origin_x = min(min_x, self.global_origin_x)
        new_origin_y = min(min_y, self.global_origin_y)
        new_width = int((max_x - new_origin_x) / self.global_resolution) + 1
        new_height = int((max_y - new_origin_y) / self.global_resolution) + 1

        # Create a new global costmap grid
        new_global_grid = np.zeros((new_height, new_width), dtype=np.int8)

        # Calculate the offset for the existing data
        offset_x = int((self.global_origin_x - new_origin_x) / self.global_resolution)
        offset_y = int((self.global_origin_y - new_origin_y) / self.global_resolution)

        # Copy the existing data to the new grid
        new_global_grid[offset_y : offset_y + self.global_height, offset_x : offset_x + self.global_width] = (
            self.global_grid
        )

        # Update the global costmap properties
        self.global_origin_x = new_origin_x
        self.global_origin_y = new_origin_y
        self.global_width = new_width
        self.global_height = new_height
        self.global_grid = new_global_grid

    def publish_global_costmap(self):
        # Create an OccupancyGrid message for the global costmap
        global_msg = OccupancyGrid()
        global_msg.header.stamp = self.get_clock().now().to_msg()
        global_msg.header.frame_id = "map"  # Global frame
        global_msg.info.resolution = self.global_resolution
        global_msg.info.width = self.global_width
        global_msg.info.height = self.global_height
        global_msg.info.origin.position.x = self.global_origin_x
        global_msg.info.origin.position.y = self.global_origin_y
        global_msg.data = self.global_grid.flatten().tolist()

        # Publish the global costmap
        self.publisher.publish(global_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalCostmapBuilder()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
