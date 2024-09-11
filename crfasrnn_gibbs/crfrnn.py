"""
MIT License

Copyright (c) 2019 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn

from crfasrnn_gibbs.filters_plattice import SpatialFilter3D, BilateralFilter3D
from crfasrnn_gibbs.filters import SpatialFilter, BilateralFilter
from crfasrnn_gibbs.params import DenseCRFParams


class CrfRnn(nn.Module):
    """
    PyTorch implementation of the CRF-RNN module described in the paper:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015 (https://arxiv.org/abs/1502.03240).
    """

    def __init__(self, num_labels, num_iterations=5, image=None, crf_init_params=None):
        """
        Create a new instance of the CRF-RNN layer.

        Args:
            num_labels:         Number of semantic labels in the dataset
            num_iterations:     Number of mean-field iterations to perform
            crf_init_params:    CRF initialization parameters
        """
        super(CrfRnn, self).__init__()

        if crf_init_params is None:
            crf_init_params = DenseCRFParams(image)

        self.params = crf_init_params
        self.num_iterations = num_iterations

        self._softmax = torch.nn.Softmax(dim=0)

        self.num_labels = num_labels

        # --------------------------------------------------------------------------------------------
        # --------------------------------- Trainable Parameters -------------------------------------
        # --------------------------------------------------------------------------------------------

        # Spatial kernel weights
        self.spatial_ker_weights = nn.Parameter(
            crf_init_params.spatial_ker_weight
            * torch.eye(num_labels, dtype=torch.float32)
        )

        # Bilateral kernel weights
        self.bilateral_ker_weights = nn.Parameter(
            crf_init_params.bilateral_ker_weight
            * torch.eye(num_labels, dtype=torch.float32)
        )

        # Compatibility transform matrix
        self.compatibility_matrix = nn.Parameter(
            torch.eye(num_labels, dtype=torch.float32)
        )

        #self.weight_dist = nn.Parameter(torch.tensor([1.0,1.0]))

    def forward(self, image, logits, unary):
        """
        Perform CRF inference for 3D data.
        Notes: unary (unary potential), msg_passing_out (pairwise potential) is all in terms of log p(h|x).
                Thus, softmax does perform the correct operation in terms of probability. 
                Thus, addition should be used for unary + msg_passing_out

        Args:
            image:  Tensor of shape (3, D, H, W) containing the RGBD or grayscale volume
            logits: Tensor of shape (num_classes, D, H, W) containing the unary logits
        Returns:
            log-Q distributions (logits) after CRF inference
        """
        if logits.shape[0] != 1:
            raise ValueError("Only batch size 1 is currently supported!")

        # Adjust the shapes for 3D data
        image = image.squeeze(0)  # Remove the first dimension if it's of size 1
        logits = logits.squeeze(0)  # Remove the first dimension if it's of size 1
        unary = unary.squeeze(0)  # Remove the first dimension if it's of size 1

        spatial_filter = SpatialFilter(image, gamma=self.params.gamma)
        bilateral_filter = BilateralFilter(image, alpha=self.params.alpha, beta=self.params.beta)

        _, d, h, w = image.shape
        cur_logits = logits
        
        q_values = self._softmax(cur_logits) # crfasrnn paper says U = -unary, double check

        for _ in range(self.num_iterations):

            # Spatial filtering
            spatial_out = torch.mm(
                self.spatial_ker_weights,
                spatial_filter.apply(q_values).view(self.num_labels, -1),
            )

            # Bilateral filtering
            bilateral_out = torch.mm(
                self.bilateral_ker_weights,
                bilateral_filter.apply(q_values).view(self.num_labels, -1),
            )

            # Compatibility transform
            msg_passing_out = spatial_out + bilateral_out 

            msg_passing_out = torch.mm(self.compatibility_matrix, msg_passing_out).view(
                self.num_labels, d, h, w
            )

            # Adding unary potentials back
            cur_logits = unary + msg_passing_out 
                         # is + the right operation here? I think it's - based on the formula, but original code does +. 
                         #       so must be that msg_passing_out is a -potential, similar to unary

            # Normalization
            q_values = self._softmax(cur_logits) # crfasrnn paper says U = -unary, double check

        q_hat = torch.unsqueeze(self._softmax(cur_logits), 0)
        return q_hat
