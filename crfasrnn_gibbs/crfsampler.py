import torch
import torch.nn as nn
import torch.nn.functional as F

class CRFSampler:
    def __init__(self, theta_alpha, theta_beta, theta_gamma, w1, w2, burn_in, num_samples):
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.w1 = w1
        self.w2 = w2
        self.burn_in = burn_in
        self.num_samples = num_samples

    def generate_coordinates(self, shape):
        """
        Generate coordinates for the given shape.
        
        :param shape: Tuple representing the shape of the 3D data (depth, height, width).
        :return: A tensor of coordinates.
        """
        depth, height, width = shape
        z, y, x = torch.meshgrid(torch.arange(depth), torch.arange(height), torch.arange(width))
        coordinates = torch.stack([z, y, x], dim=-1).float()
        return coordinates

    def combined_kernel(self, f_i, f_j, ell_i, ell_j, x_i, x_j):
        term1 = self.w1[1] * torch.exp(-torch.norm(ell_i - ell_j)**2 / (2 * self.theta_alpha**2) - (x_i - x_j)**2 / (2 * self.theta_beta**2))
        term2 = self.w2[1] * torch.exp(-torch.norm(ell_i - ell_j)**2 / (2 * self.theta_gamma**2))
        return term1 + term2

    def psi_p(self, h_i, h_j, x_i, x_j, ell_i, ell_j, weights):
        mu_value = 1 if h_i == h_j else -1
        pairwise_sum = 0
        for m in range(len(weights)):
            kernel_value = self.combined_kernel(x_i, x_j, ell_i, ell_j, x_i, x_j)
            pairwise_sum += weights[m] * kernel_value
        return mu_value * pairwise_sum

    def compute_potential(self, labels, label_index, features, coordinates, unary):
        potential = unary.clone()
        weights = [1.0] # if we have more kernels, change this weight vector

        depth, height, width = labels.shape[-3:]
        z, y, x = label_index
        
        ell_i = coordinates[z, y, x]
        x_i = features[z, y, x]
        h_i = labels[z, y, x]
        
        ell_j = coordinates.view(-1, 3)
        x_j = features.view(-1)
        h_j = labels.view(-1)
        
        mu_value = (h_i == h_j).float() * 2 - 1
        
        term1 = self.w1[1] * torch.exp(-torch.norm(ell_i - ell_j, dim=-1)**2 / (2 * self.theta_alpha**2) - (x_i - x_j)**2 / (2 * self.theta_beta**2))
        term2 = self.w2[1] * torch.exp(-torch.norm(ell_i - ell_j, dim=-1)**2 / (2 * self.theta_gamma**2))
        
        pairwise_potential = mu_value * (term1 + term2)
        
        pairwise_potential = pairwise_potential.view(depth, height, width)
        mask = (coordinates[..., 0] < z) & (coordinates[..., 1] < y) & (coordinates[..., 2] < x)
        
        potential += (pairwise_potential * mask.float()).sum(dim=0)
        
        return potential[z, y, x]
        
    def sample_conditional(self, labels, label_index, features, coordinates, unary):
        potential = self.compute_potential(labels, label_index, features, coordinates, unary)
        logits_h1 = potential + unary[label_index]
        logits_h0 = -potential + unary[label_index]

        probabilities_h1 = torch.sigmoid(logits_h1)
        probabilities_h0 = torch.sigmoid(logits_h0)

        probabilities = torch.stack([probabilities_h0, probabilities_h1])
        probabilities /= probabilities.sum(dim=0)

        new_state = torch.multinomial(probabilities, 1).item()
        return new_state

    def gibbs_sampler_3d(self, initial_labels, features, unary):
        features = features[0, 0]
        unary = unary[0, 1]
        print(initial_labels.shape, features.shape, unary.shape)  # (2, 30, 30, 30) torch.Size([1, 1, 30, 30, 30]) torch.Size([1, 2, 30, 30, 30])

        coordinates = self.generate_coordinates(initial_labels.shape[-3:])
        labels = torch.tensor(initial_labels, dtype=torch.float32).clone()
        depth, height, width = labels.shape[-3:]

        accumulated_labels = torch.zeros_like(labels, dtype=torch.float32)
        total_iterations = self.burn_in + self.num_samples

        for iteration in range(total_iterations):
            for z in range(depth):
                for y in range(height):
                    for x in range(width):
                        print(f'\rIteration: {iteration + 1}/{total_iterations}, Position: ({x}, {y}, {z})', end=' '*10)
                        label_index = (z, y, x)
                        new_state = self.sample_conditional(labels, label_index, features, coordinates, unary)
                        labels[z, y, x] = new_state
            
            # Accumulate labels after burn-in period
            if iteration >= self.burn_in:
                accumulated_labels += labels
            
            # Print proportion of 1s in the current labels
            proportion_of_1s = (labels == 1).float().mean().item()
            print(f'\rIteration: {iteration + 1}/{total_iterations}, Proportion of 1s: {proportion_of_1s:.4f}', end=' '*10)

        # Average the accumulated labels over the sampling iterations
        averaged_labels = accumulated_labels / self.num_samples
        
        return averaged_labels

