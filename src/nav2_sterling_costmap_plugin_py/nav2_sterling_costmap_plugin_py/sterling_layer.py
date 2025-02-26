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
        # Define the bounds to cover the area where the triangle will be drawn
        min_x[0] = robot_x - 2.0  # 2 meters behind the robot
        min_y[0] = robot_y - 2.0  # 2 meters to the left of the robot
        max_x[0] = robot_x + 4.0  # 4 meters in front of the robot
        max_y[0] = robot_y + 2.0  # 2 meters to the right of the robot

        # Save the robot's position and orientation for use in update_costs
        self.last_x = robot_x
        self.last_y = robot_y
        self.last_yaw = robot_yaw

        return (min_x[0], min_y[0], max_x[0], max_y[0])

    def update_costs(self, master_grid, min_i, min_j, max_i, max_j):
        """
        Update the costmap with a triangle in front of the robot.

        Args:
            master_grid (Costmap2D): The master costmap grid.
            min_i (int): Minimum x index in the costmap.
            min_j (int): Minimum y index in the costmap.
            max_i (int): Maximum x index in the costmap.
            max_j (int): Maximum y index in the costmap.
        """
        # Define the triangle vertices relative to the robot's position and orientation
        triangle_vertices = [
            (1.0, 0.0),  # 1 meter in front of the robot
            (2.0, -1.0),  # 2 meters in front and 1 meter to the left
            (2.0, 1.0)    # 2 meters in front and 1 meter to the right
        ]

        # Rotate and translate the triangle vertices based on the robot's yaw
        cos_yaw = np.cos(self.last_yaw)
        sin_yaw = np.sin(self.last_yaw)

        transformed_vertices = []
        for (x, y) in triangle_vertices:
            # Rotate the vertices
            x_rot = x * cos_yaw - y * sin_yaw
            y_rot = x * sin_yaw + y * cos_yaw

            # Translate the vertices to the robot's position
            x_trans = x_rot + self.last_x
            y_trans = y_rot + self.last_y

            # Convert to costmap coordinates
            i = int((x_trans - master_grid.origin_x) / master_grid.resolution)
            j = int((y_trans - master_grid.origin_y) / master_grid.resolution)

            transformed_vertices.append((i, j))

        # Draw the triangle on the costmap
        self.draw_triangle(master_grid, transformed_vertices)

        return master_grid

    def draw_triangle(self, master_grid, vertices):
        """
        Draw a triangle on the costmap.

        Args:
            master_grid (Costmap2D): The master costmap grid.
            vertices (list): List of (i, j) vertices of the triangle.
        """
        # Get the bounding box of the triangle
        i_coords = [v[0] for v in vertices]
        j_coords = [v[1] for v in vertices]
        min_i = min(i_coords)
        max_i = max(i_coords)
        min_j = min(j_coords)
        max_j = max(j_coords)

        # Iterate over the bounding box and check if each cell is inside the triangle
        for j in range(min_j, max_j + 1):
            for i in range(min_i, max_i + 1):
                if self.is_point_in_triangle((i, j), vertices):
                    master_grid.set_cost(i, j, 100)  # Set the cost to 100 (occupied)

    def is_point_in_triangle(self, point, vertices):
        """
        Check if a point is inside a triangle.

        Args:
            point (tuple): (i, j) coordinates of the point.
            vertices (list): List of (i, j) vertices of the triangle.

        Returns:
            bool: True if the point is inside the triangle, False otherwise.
        """
        def cross_product(a, b):
            return a[0] * b[1] - a[1] * b[0]

        def same_side(p1, p2, a, b):
            cp1 = cross_product((b[0] - a[0], b[1] - a[1]), (p1[0] - a[0], p1[1] - a[1]))
            cp2 = cross_product((b[0] - a[0], b[1] - a[1]), (p2[0] - a[0], p2[1] - a[1]))
            return cp1 * cp2 >= 0

        a, b, c = vertices
        p = point

        return (same_side(p, a, b, c) and
                same_side(p, b, a, c) and
                same_side(p, c, a, b))