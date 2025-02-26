import numpy as np
from nav2_costmap_2d.costmap_layer import CostmapLayer
from nav2_costmap_2d.costmap import Costmap2D

class SterlingLayer(CostmapLayer):
    def __init__(self):
        super().__init__()
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_yaw = 0.0

    def on_initialize(self):
        """
        Initialize the plugin.
        """
        self.get_logger().info("SterlingLayer initialized")

    def update_bounds(self, robot_x, robot_y, robot_yaw, min_x, min_y, max_x, max_y):
        """
        Update the bounds of the costmap.

        Args:
            robot_x (float): Robot's x position in the global frame.
            robot_y (float): Robot's y position in the global frame.
            robot_yaw (float): Robot's yaw (heading) in radians.
            min_x (list): Minimum x bound (output).
            min_y (list): Minimum y bound (output).
            max_x (list): Maximum x bound (output).
            max_y (list): Maximum y bound (output).
        """
        # Example: Update bounds to cover a 6x6 meter area around the robot
        min_x[0] = robot_x - 3.0
        min_y[0] = robot_y - 3.0
        max_x[0] = robot_x + 3.0
        max_y[0] = robot_y + 3.0

        # Save the robot's position for use in update_costs
        self.last_x = robot_x
        self.last_y = robot_y
        self.last_yaw = robot_yaw

        return (min_x[0], min_y[0], max_x[0], max_y[0])

    def update_costs(self, master_grid, min_i, min_j, max_i, max_j):
        """
        Update the costmap values.

        Args:
            master_grid (Costmap2D): The master costmap grid.
            min_i (int): Minimum x index in the costmap.
            min_j (int): Minimum y index in the costmap.
            max_i (int): Maximum x index in the costmap.
            max_j (int): Maximum y index in the costmap.
        """
        # Example: Set a gradient cost pattern around the robot
        for j in range(min_j, max_j):
            for i in range(min_i, max_i):
                # Calculate the distance from the robot
                dx = (i - min_i) * master_grid.resolution
                dy = (j - min_j) * master_grid.resolution
                distance = np.sqrt(dx**2 + dy**2)

                # Set the cost based on distance (e.g., higher cost closer to the robot)
                cost = int(100 * (1.0 - distance / 3.0))  # Scale cost from 100 to 0
                cost = max(0, min(100, cost))  # Clamp cost to [0, 100]

                # Update the costmap
                master_grid.set_cost(i, j, cost)

        return master_grid