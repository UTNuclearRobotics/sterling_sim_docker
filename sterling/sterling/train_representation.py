import torch.nn as nn
import torch.nn.functional as F

from sterling.vicreg_loss import VICRegLoss
from sterling.visual_encoder_model import VisualEncoderModel


class SterlingPaternRepresentation(nn.Module):
    def __init__(self, device):
        super(SterlingPaternRepresentation, self).__init__()
        self.device = device
        self.latent_size = 128
        self.rep_size = self.latent_size
        self.visual_encoder = VisualEncoderModel(self.latent_size)
        #self.proprioceptive_encoder = ProprioceptionModel(self.latent_size)
        self.projector = nn.Sequential(
            nn.Linear(self.rep_size, self.latent_size),
            nn.PReLU(),
            nn.Linear(self.latent_size, self.latent_size),
        )

        self.vicreg_loss = VICRegLoss()
        self.l1_coeff = 0.5

    def forward(self, patch1, patch2):
        """
        Args:
            patch1 (torch.Tensor): First patch image of shape (3, 128, 128)
            patch2 (torch.Tensor): Second patch image of shape (3, 128, 128)
        """

        # Encode visual patches
        patch1 = patch1.to(self.device)
        patch2 = patch2.to(self.device)
        v_encoded_1 = self.visual_encoder(patch1)
        v_encoded_1 = F.normalize(v_encoded_1, dim=-1)
        v_encoded_2 = self.visual_encoder(patch2)
        v_encoded_2 = F.normalize(v_encoded_2, dim=-1)

        #i_encoded = self.proprioceptive_encoder(inertial_data.float())

        # Project encoded representations to latent space
        zv1 = self.projector(v_encoded_1)
        zv2 = self.projector(v_encoded_2)
        #zi = self.projector(i_encoded)

        return zv1, zv2, v_encoded_1, v_encoded_2
    
    def encode_single_patch(self, patch):
        """
        Encode a single patch and return its representation vector.
        Args:
            patch (torch.Tensor): Single patch image of shape (1, 3, H, W).
        Returns:
            torch.Tensor: Encoded and normalized representation vector.
        """
        # Ensure the input is on the correct device
        patch = patch.to(self.device)

        # Encode the patch
        v_encoded = self.visual_encoder(patch)
        v_encoded = F.normalize(v_encoded, dim=-1)  # Normalize the representation vector
        return v_encoded
    
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step on the batch of data.
        Args:
            batch (tuple): A tuple containing the input data.
            batch_idx (int): The index of the current batch.
        Returns:
            torch.Tensor: The computed loss value.
        """
        patch1, patch2 = batch
        zv1, zv2, _, _ = self.forward(patch1, patch2)

        # Compute VICReg loss
        loss_vpt_inv = self.vicreg_loss(zv1, zv2)
        #loss_vi = 0.5 * self.vicreg_loss(zv1,zi) + 0.5 * self.vicreg_loss(zv2,zi)

        #loss = self.l1_coeff * loss_vpt_inv + (1.0-self.l1_coeff) * loss_vi

        return loss_vpt_inv