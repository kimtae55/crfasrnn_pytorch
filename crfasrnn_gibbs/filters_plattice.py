from crfasrnn_gibbs.plattice import PermutohedralLattice
import torch
import numpy as np
from abc import ABC, abstractmethod

def _spatial_features_3d(image, sigma):
    D, H, W = image.size()[-3:]
    device = image.device

    depth = torch.arange(D, dtype=torch.float32, device=device).view(D, 1, 1) / sigma
    height = torch.arange(H, dtype=torch.float32, device=device).view(1, H, 1) / sigma
    width = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, W) / sigma

    dd = depth.repeat(1, H, W)
    hh = height.repeat(D, 1, W)
    ww = width.repeat(D, H, 1)

    print('_spatial_features_3d > torch.stack([dd, hh, ww], dim=-1): ',torch.stack([dd, hh, ww], dim=-1))
    return torch.stack([dd, hh, ww], dim=-1)  # Shape: [D, H, W, 3]

class AbstractFilter3D:
    def __init__(self, image):
        self.image = image
        self.features = self._calc_features(image)
        D, H, W, _ = self.features.shape
        self.num_elements = D * H * W
        self.value_dim = image.size(0)  # Assuming image is in [C, D, H, W] format

    def apply(self, input_):
        feature_dim = self.features.shape[-1]
        output_np = PermutohedralLattice.filter_3d(input_.detach().numpy(), self.features.detach().numpy())
        output = torch.tensor(output_np, device=input_.device, dtype=input_.dtype)
        norm = 1.0 / (self._calc_norm() + torch.finfo(input_.dtype).eps)
        return output * norm

    @abstractmethod
    def _calc_features(self, image):
        pass

    def _calc_norm(self):
        all_ones = torch.ones_like(self.image)
        norm_np = PermutohedralLattice.filter_3d(all_ones.detach().numpy(), self.features.detach().numpy())
        norm = torch.tensor(norm_np, device=self.image.device, dtype=self.image.dtype)
        return norm

class SpatialFilter3D(AbstractFilter3D):
    def __init__(self, image, gamma):
        self.gamma = gamma
        super().__init__(image)

    def _calc_features(self, image):
        return _spatial_features_3d(image, self.gamma)

class BilateralFilter3D(AbstractFilter3D):
    def __init__(self, image, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        super().__init__(image)

    def _calc_features(self, image):
        spatial_features = _spatial_features_3d(image, self.alpha)
        color_features = (image.permute(1, 2, 3, 0) / float(self.beta))  # Adjusting for channel last order in 3D
        print(spatial_features.shape, color_features.shape)
        return torch.cat([spatial_features, color_features], dim=-1)  # Concatenate along the feature dimension

