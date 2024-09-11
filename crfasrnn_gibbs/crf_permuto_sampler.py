import torch
import torch.nn as nn
import torch.nn.functional as F
from crfasrnn_gibbs.filters_plattice import SpatialFilter3D, BilateralFilter3D
from crfasrnn_gibbs.filters import SpatialFilter, BilateralFilter
from crfasrnn_gibbs.params import DenseCRFParams
import multiprocessing
import os 

class CRFPermutoSampler:
    def __init__(self, burn_in, num_samples, trained_weights=None, image=None):
        num_labels = 2 # binary

        self.params = DenseCRFParams(image)
        self.burn_in = burn_in
        self.num_samples = num_samples
        # Spatial kernel weights
        self.spatial_ker_weights = trained_weights[0]

        # Bilateral kernel weights
        self.bilateral_ker_weights = trained_weights[1]

        # Compatibility transform matrix
        self.compatibility_matrix = trained_weights[2]


    def compute_potential(self, labels, label_index, unary):
        z, y, x = label_index
        
        # Initialize potentials as zeros
        potential_label_0 = torch.zeros_like(unary[0])
        potential_label_1 = torch.zeros_like(unary[1])

        q_values = torch.stack([labels, 1 - labels], dim=0)

        # Define the computation of spatial and bilateral message passing outputs
        spatial_out = torch.mm(
            self.spatial_ker_weights,
            self.spatial_filter.apply(q_values).view(2, -1)  # Adjusting view to handle both labels
        )
        bilateral_out = torch.mm(
            self.bilateral_ker_weights,
            self.bilateral_filter.apply(q_values).view(2, -1)  # Adjusting view to handle both labels
        )

        # Sum spatial and bilateral outputs
        msg_passing_out = spatial_out + bilateral_out
        msg_passing_out = msg_passing_out.view(2, *labels.shape[-3:])

        # Update the potential labels with the message passing output
        potential_label_0[z, y, x] += msg_passing_out[0, z, y, x]  # For label 0
        potential_label_1[z, y, x] += msg_passing_out[1, z, y, x]  # For label 1

        return potential_label_0[z, y, x], potential_label_1[z, y, x]

    def sample_conditional(self, labels, label_index, features, unary):
        z, y, x = label_index

        # Compute potentials for both label 0 and label 1
        potential_0, potential_1 = self.compute_potential(labels, label_index, unary)

        # Stack potentials for label 0 and label 1
        logits = torch.stack([potential_0 + unary[0, z, y, x], potential_1 + unary[1, z, y, x]])

        # Apply softmax to obtain normalized probabilities
        probabilities = torch.softmax(logits, dim=0)

        # Sample a new state based on the computed probabilities
        new_state = torch.multinomial(probabilities, 1).item()
        return new_state

    def gibbs_sampler_3d(self, initial_labels, image, unary):
        features = image[0, 0] 
        unary = unary[0]

        image = image.squeeze(0)  # Remove the first dimension if it's of size 1
        self.spatial_filter = SpatialFilter(image, gamma=self.params.gamma)
        self.bilateral_filter = BilateralFilter(image, alpha=self.params.alpha, beta=self.params.beta)

        labels = torch.tensor(initial_labels, dtype=torch.float32).clone()
        depth, height, width = labels.shape[-3:]

        accumulated_labels = torch.zeros_like(labels, dtype=torch.float32)
        total_iterations = self.burn_in + self.num_samples

        for iteration in range(total_iterations):
            print(f'\rIteration: {iteration + 1}/{total_iterations}', end=' '*10)

            for z in range(depth):
                for y in range(height):
                    for x in range(width):
                        label_index = (z, y, x)
                        new_state = self.sample_conditional(labels, label_index, features, unary)
                        labels[z, y, x] = new_state
            
            # Print proportion of 1s in the current labels
            proportion_of_1s = (labels == 1).float().mean().item()
            print(f'\nProportion of 1s: {proportion_of_1s:.4f}')

            # Accumulate labels after burn-in period
            if iteration >= self.burn_in:
                accumulated_labels += labels
            

        # Average the accumulated labels over the sampling iterations
        averaged_labels = accumulated_labels / self.num_samples
        
        return averaged_labels


    def sample_chunk(self, chunk_data):
        # Extract the chunk data
        labels, features, unary = chunk_data
        depth, height, width = labels.shape[-3:]

        # Print process ID for debugging
        print(f"Processing in process ID: {multiprocessing.current_process()}")

        # Perform a single Gibbs sampling iteration
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    label_index = (z, y, x)
                    new_state = self.sample_conditional(labels, label_index, features, unary)
                    labels[z, y, x] = new_state

        return labels

    def gibbs_sampler_3d_parallel(self, initial_labels, image, unary):
        features = image[0, 0]
        unary = unary[0]

        # Prepare filters, etc.
        image = image.squeeze(0)
        self.spatial_filter = SpatialFilter(image, gamma=self.params.gamma)
        self.bilateral_filter = BilateralFilter(image, alpha=self.params.alpha, beta=self.params.beta)

        labels = torch.tensor(initial_labels, dtype=torch.float32).clone()
        accumulated_labels = torch.zeros_like(labels, dtype=torch.float32)
        total_iterations = self.burn_in + self.num_samples

        labels = labels.detach()
        depth, height, width = labels.shape[-3:]

        # Sequential processing during burn-in period
        print("Starting burn-in period...")
        for iteration in range(self.burn_in):
            print(f'\rIteration: {iteration + 1}/{total_iterations}', end=' ' * 10)
            # Use sample_chunk function for single iteration during burn-in
            labels = self.sample_chunk((labels, features, unary))

            # Print proportion of 1s in the labels after each burn-in iteration
            proportion_of_1s = (labels == 1).float().mean().item()
            print(f'\nProportion of 1s after burn-in iteration {iteration + 1}: {proportion_of_1s:.4f}')

        # After burn-in, use parallel processing
        remaining_iterations = total_iterations - self.burn_in
        # can use cpu_count(), i just define it as 10 for now
        num_workers = min(10, multiprocessing.cpu_count())
        print('num_processes: ', num_workers)

        print("Starting parallel processing...")
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Run iterations in parallel, distributing the remaining iterations across workers
            chunk_data = [(labels.clone().detach(), features.detach(), unary.detach()) for _ in range(remaining_iterations)]
            results = pool.map(self.sample_chunk, chunk_data)

        # Accumulate results from parallel processing
        for result in results:
            accumulated_labels += result

        # Average the accumulated labels over the sampling iterations
        averaged_labels = accumulated_labels / self.num_samples

        return averaged_labels


        