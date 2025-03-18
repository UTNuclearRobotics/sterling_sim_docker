import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from sterling.proprioception_model import ProprioceptionModel
from sterling.visual_encoder_model import VisualEncoderModel


class PaternPreAdaptation(nn.Module):
    def __init__(self, device, pretrained_weights_path=None, latent_size=128):
        super(PaternPreAdaptation, self).__init__()
        self.device = device
        self.latent_size = latent_size  # Fixed at 128D

        # Initialize encoders
        self.visual_encoder = VisualEncoderModel(latent_size=self.latent_size)
        self.proprioceptive_encoder = ProprioceptionModel(latent_size=self.latent_size)

        # Load pre-trained weights if provided
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            weight_files = {
                "visual_encoder": "fvis.pt",
                "proprioceptive_encoder": "fpro.pt",
                "uvis": "uvis.pt",
                "upro": "upro.pt",
                "cost_head": "cost_head.pt",
            }
            all_files_exist = all(
                os.path.exists(os.path.join(pretrained_weights_path, file_name)) for file_name in weight_files.values()
            )
            if all_files_exist:
                for submodule_name, file_name in weight_files.items():
                    file_path = os.path.join(pretrained_weights_path, file_name)
                    state_dict = torch.load(file_path, weights_only=True, map_location=device)
                    submodule = getattr(self, submodule_name)
                    submodule.load_state_dict(state_dict)
                    print(f"Loaded {submodule_name} weights from {file_path} for fine-tuning")
            else:
                print(
                    f"Warning: Not all required weight files found in {pretrained_weights_path}. Initializing from scratch."
                )
        else:
            print(f"No pre-trained weights directory found at {pretrained_weights_path}. Initializing from scratch.")

        # Utility functions (2-layer MLP on 128D vectors with scaling to 0-255)
        self.uvis = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size // 2),
            nn.ReLU(),
            nn.Linear(self.latent_size // 2, 1),
            nn.ReLU(),
        )
        self.upro = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size // 2),
            nn.ReLU(),
            nn.Linear(self.latent_size // 2, 1),
            nn.ReLU(),
        )

        self.cost_head = nn.Sequential(
            nn.Linear(1, 128),  # Increased capacity
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

        # Initialize weights and biases to encourage positive outputs
        nn.init.kaiming_normal_(self.cost_head[0].weight, mode="fan_in", nonlinearity="relu")
        self.cost_head[0].bias.data.fill_(1.0)  # Positive bias to avoid ReLU zeroing out
        nn.init.kaiming_normal_(self.cost_head[2].weight, mode="fan_in", nonlinearity="relu")
        self.cost_head[2].bias.data.fill_(1.0)
        nn.init.kaiming_normal_(self.cost_head[4].weight, mode="fan_in", nonlinearity="relu")
        self.cost_head[4].bias.data.fill_(1.0)

        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)

    def forward(self, patches, inertial=None):
        patches = patches.to(self.device)
        phi_vis = self.visual_encoder(patches)
        uvis_pred = self.uvis(phi_vis)

        # Optionally process inertial data if provided
        if inertial is not None:
            inertial = inertial.to(self.device)
            phi_pro = self.proprioceptive_encoder(inertial.float())
            upro_pred = self.upro(phi_pro)
        else:
            phi_pro = torch.zeros_like(phi_vis)  # Dummy for consistency
            upro_pred = torch.zeros_like(uvis_pred)  # Dummy for consistency

        # Use only uvis_pred for final cost
        final_cost = self.cost_head(uvis_pred)

        return phi_vis, phi_pro, uvis_pred, upro_pred, final_cost

    def training_step(self, batch, batch_idx):
        patches, inertial, terrain_labels, preferences = batch
        preferences = preferences.to(self.device).float()
        scaled_preferences = preferences * 25.5

        phi_vis, phi_pro, uvis_pred, upro_pred, final_cost = self.forward(patches, inertial)

        # Scale predictions to 0-255
        uvis_pred = uvis_pred * 255.0 / uvis_pred.max() if uvis_pred.max() > 0 else uvis_pred

        terrain_labels_tensor = torch.tensor(
            [hash(label) for label in terrain_labels], dtype=torch.long, device=self.device
        )
        batch_size = len(terrain_labels)
        labels_expanded = terrain_labels_tensor.unsqueeze(1)
        pos_mask = (labels_expanded == labels_expanded.t()) & ~torch.eye(
            batch_size, dtype=torch.bool, device=self.device
        )
        neg_mask = labels_expanded != labels_expanded.t()

        pos_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        neg_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            pos_candidates = pos_mask[i].nonzero(as_tuple=False).flatten()
            neg_candidates = neg_mask[i].nonzero(as_tuple=False).flatten()
            pos_indices[i] = (
                pos_candidates[torch.randint(0, len(pos_candidates), (1,), device=self.device)]
                if len(pos_candidates) > 0
                else i
            )
            neg_indices[i] = (
                neg_candidates[torch.randint(0, len(neg_candidates), (1,), device=self.device)]
                if len(neg_candidates) > 0
                else i
            )

        vis_loss = self.triplet_loss(phi_vis, phi_vis[pos_indices], phi_vis[neg_indices])
        pro_loss = self.triplet_loss(phi_pro, phi_pro[pos_indices], phi_pro[neg_indices])

        pref_diff = preferences.unsqueeze(1) - preferences.unsqueeze(0)
        pred_diff = uvis_pred.unsqueeze(1) - uvis_pred.unsqueeze(0)
        ranking_mask = pref_diff > 0
        ranking_loss = (
            F.relu(1.0 - (pred_diff / 25.5)[ranking_mask]).mean()
            if ranking_mask.any()
            else torch.tensor(0.0, device=self.device)
        )

        modality_mse_loss = F.mse_loss(uvis_pred.detach(), upro_pred)

        cost_loss = F.mse_loss(final_cost, scaled_preferences)

        total_loss = 1.0 * (vis_loss + pro_loss) + 0.5 * ranking_loss + 0.5 * modality_mse_loss + 1.0 * cost_loss

        # print(f"Train Batch {batch_idx}: vis_loss={vis_loss.item():.4f}, pro_loss={pro_loss.item():.4f}, "
        #      f"ranking_loss={ranking_loss.item():.4f}, modality_mse_loss={modality_mse_loss.item():.4f}, "
        #      f"cost_loss={cost_loss.item():.4f}, total_loss={total_loss.item():.4f}")
        print(f"uvis_pred range: {uvis_pred.min().item():.4f} to {uvis_pred.max().item():.4f}")
        print(f"final_cost range: {final_cost.min().item():.4f} to {final_cost.max().item():.4f}")
        print(
            f"scaled_preferences range: {scaled_preferences.min().item():.4f} to {scaled_preferences.max().item():.4f}"
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        patches, inertial, terrain_labels, preferences = batch
        preferences = preferences.to(self.device).float()
        scaled_preferences = preferences * 25.5

        phi_vis, phi_pro, uvis_pred, upro_pred, final_cost = self.forward(patches, inertial)

        # Scale predictions to 0-255
        uvis_pred = uvis_pred * 255.0 / uvis_pred.max() if uvis_pred.max() > 0 else uvis_pred
        final_cost = final_cost * 255.0 / final_cost.max() if final_cost.max() > 0 else final_cost

        terrain_labels_tensor = torch.tensor(
            [hash(label) for label in terrain_labels], dtype=torch.long, device=self.device
        )
        batch_size = len(terrain_labels)
        labels_expanded = terrain_labels_tensor.unsqueeze(1)
        pos_mask = (labels_expanded == labels_expanded.t()) & ~torch.eye(
            batch_size, dtype=torch.bool, device=self.device
        )
        neg_mask = labels_expanded != labels_expanded.t()
        pos_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        neg_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            pos_candidates = pos_mask[i].nonzero(as_tuple=False).flatten()
            neg_candidates = neg_mask[i].nonzero(as_tuple=False).flatten()
            pos_indices[i] = (
                pos_candidates[torch.randint(0, len(pos_candidates), (1,), device=self.device)]
                if len(pos_candidates) > 0
                else i
            )
            neg_indices[i] = (
                neg_candidates[torch.randint(0, len(neg_candidates), (1,), device=self.device)]
                if len(neg_candidates) > 0
                else i
            )
        vis_loss = self.triplet_loss(phi_vis, phi_vis[pos_indices], phi_vis[neg_indices])
        pro_loss = self.triplet_loss(phi_pro, phi_pro[pos_indices], phi_pro[neg_indices])

        pref_diff = preferences.unsqueeze(1) - preferences.unsqueeze(0)
        pred_diff = uvis_pred.unsqueeze(1) - uvis_pred.unsqueeze(0)
        ranking_mask = pref_diff > 0
        ranking_loss = (
            F.relu(1.0 - (pred_diff / 25.5)[ranking_mask]).mean()
            if ranking_mask.any()
            else torch.tensor(0.0, device=self.device)
        )

        modality_mse_loss = F.mse_loss(uvis_pred.detach(), upro_pred)

        cost_loss = F.mse_loss(final_cost, scaled_preferences)

        total_loss = 1.0 * (vis_loss + pro_loss) + 0.5 * ranking_loss + 0.5 * modality_mse_loss + 1.0 * cost_loss
        return total_loss


def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, device):
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model.training_step(batch, batch_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                val_loss = model.validation_step(batch, batch_idx)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        scheduler.step()
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")