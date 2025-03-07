import os

import joblib
import numpy as np
import torch
from sklearn.preprocessing import normalize

from sterling.train_representation import SterlingPaternRepresentation


class BEVCostmap:
    """
    An overview of the cost inference process for local planning at deployment.
    """

    def __init__(self, viz_encoder_path, kmeans_path, preferences):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load visual encoder model weights
        self.sterling = SterlingPaternRepresentation(self.device).to(self.device)
        if not os.path.exists(viz_encoder_path):
            raise FileNotFoundError(f"Model file not found at: {viz_encoder_path}")
        self.sterling.load_state_dict(
            torch.load(viz_encoder_path, weights_only=True, map_location=torch.device(self.device))
        )

        # Load K-means model
        if not os.path.exists(kmeans_path):
            raise FileNotFoundError(f"K-means model not found at {kmeans_path}")
        self.kmeans = joblib.load(kmeans_path)

        self.preferences = preferences

    def predict_clusters(self, cells):
        """Predict clusters for a batch of cells."""
        if isinstance(cells, np.ndarray):
            cells = torch.tensor(cells, dtype=torch.float32, device=self.device)

        if len(cells.shape) == 4:  # [B, C, H, W]
            pass
        elif len(cells.shape) == 3:  # [C, H, W] -> [1, C, H, W]
            cells = cells.unsqueeze(0)

        self.sterling.eval()
        with torch.no_grad():
            representation_vectors = self.sterling.visual_encoder(cells)
            # Ensure representation_vectors is on CPU
            representations_np = representation_vectors.cpu().numpy()
            representations_np = normalize(representations_np, axis=1, norm="l2")

        return self.kmeans.predict(representations_np)

    def calculate_cell_costs(self, cells):
        """Batch process cell costs."""
        cluster_labels = self.predict_clusters(cells)
        costs = [self.preferences[label] for label in cluster_labels]
        return costs

    def BEV_to_costmap(self, bev_img, cell_size):
        """Convert BEV image to costmap while automatically marking consistent black areas."""
        height, width = bev_img.shape[:2]
        num_cells_y, num_cells_x = height // cell_size, width // cell_size

        # Determine effective dimensions that are multiples of cell_size.
        effective_height = num_cells_y * cell_size
        effective_width = num_cells_x * cell_size

        # Slice the image to the effective region.
        bev_img = bev_img[:effective_height, :effective_width]

        # Initialize costmap container.
        costmap = np.empty((num_cells_y, num_cells_x), dtype=np.int8)

        # Create mask for black regions.
        black_cells = np.zeros((num_cells_y, num_cells_x), dtype=bool)
        black_cells[-2, [0, -1]] = True  # Row -2, columns 0 and -1
        black_cells[-1, [0, 1, -2, -1]] = True  # Row -1, columns 0, 1, -2, and -1

        # Use stride tricks to extract cell views without copying data.
        channels = bev_img.shape[2]
        cell_shape = (num_cells_y, num_cells_x, cell_size, cell_size, channels)
        cell_strides = (
            bev_img.strides[0] * cell_size,
            bev_img.strides[1] * cell_size,
            bev_img.strides[0],
            bev_img.strides[1],
            bev_img.strides[2],
        )
        cells = np.lib.stride_tricks.as_strided(bev_img, shape=cell_shape, strides=cell_strides)
        
        # Rearrange to (num_cells_y, num_cells_x, channels, cell_size, cell_size)
        cells = cells.transpose(0, 1, 4, 2, 3)

        # Select only valid (non-black) cells.
        valid_cells = cells[~black_cells]

        # Calculate costs for valid cells in a single batch.
        if valid_cells.size:
            costs = self.calculate_cell_costs(valid_cells)
        else:
            costs = np.empty((0,), dtype=np.uint8)

        # Assemble costmap: assign maximum cost (-1) to unknown cells and computed costs to others.
        costmap[black_cells] = -1
        costmap[~black_cells] = costs

        return costmap
