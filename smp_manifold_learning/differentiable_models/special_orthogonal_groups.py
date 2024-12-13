# File: special_orthogonal_groups.py
# Author: Giovanni Sutanto
# Email: gsutanto@alumni.usc.edu
# Date: June 2020
# Remarks: This is an implementation of a
#          (batched) differentiable Special Orthogonal Groups (SO(n)),
#          via matrix exponentials
#          ( Shepard, et al. (2015).
#            "The Representation and Parametrization of Orthogonal Matrices".
#            https://pubs.acs.org/doi/abs/10.1021/acs.jpca.5b02015 ).
#          This is useful for an (Iterative) Orthogonal Subspace Alignment (IOSA) of
#          (potentially high-dimensional) SO(n) coordinate frames, e.g. on a manifold.
#          This is implemented in TensorFlow 2.2 because
#          at the time this code is written (June 2020),
#          PyTorch does NOT yet support differentiable matrix exponentials (expm)
#          ( https://github.com/pytorch/pytorch/issues/9983 ).
#
import torch
import numpy as np


def convert_to_skewsymm(batch_params):
    """
    Convert batch of parameters to skew-symmetric matrices.
    
    Args:
        batch_params (torch.Tensor or np.ndarray): Batch of parameters to convert
    
    Returns:
        torch.Tensor: Batch of skew-symmetric matrices
    """
    # Ensure batch_params is a torch tensor
    if not isinstance(batch_params, torch.Tensor):
        batch_params = torch.tensor(batch_params, dtype=torch.float32)
    
    N_batch = batch_params.shape[0]
    
    # Search for the skew-symmetricity dimension
    i = 2
    while (int(round((i * (i - 1)) / 2)) < batch_params.shape[1]):
        i += 1
    
    assert int(round((i * (i - 1)) / 2)) == batch_params.shape[1], \
        "Skew-symmetricity dimension is NOT found!"
    n = i

    # Create zero tensor for skew-symmetric matrices
    ret_tensor = torch.zeros(N_batch, n, n, dtype=batch_params.dtype, device=batch_params.device)
    
    # Get lower triangular indices
    ii, jj = np.tril_indices(n=n, k=-1, m=n)
    
    # Unpack the parameters and populate skew-symmetric matrices
    for k in range(N_batch):
        for i, j, vec_params in zip(ii, jj, batch_params[k]):
            ret_tensor[k, i, j] = vec_params
            ret_tensor[k, j, i] = -vec_params
    
    return ret_tensor


class SpecialOrthogonalGroups:
    def __init__(self, n, N_batch=1, rand_seed=38):
        """
        Initialize Special Orthogonal Groups.
        
        Args:
            n (int): Dimension of the orthogonal group
            N_batch (int, optional): Number of batch elements. Defaults to 1.
            rand_seed (int, optional): Random seed. Defaults to 38.
        """
        torch.manual_seed(rand_seed)
        
        self.N_batch = N_batch
        assert n >= 1
        self.n = n
        self.dim_params = int(round((n * (n - 1)) / 2))
        
        if self.dim_params > 0:
            # Initialize parameters with small random values
            self.params = torch.nn.ParameterList([
                torch.nn.Parameter(torch.normal(0, 1e-7, size=(self.dim_params,))) 
                for _ in range(self.N_batch)
            ])
        else:
            # For 1-dimensional case, create a dummy parameter list
            self.params = torch.nn.ParameterList([
                torch.nn.Parameter(torch.tensor([0.0]))
            ])

    def __call__(self):
        """
        Generate orthogonal matrices.
        
        Returns:
            torch.Tensor: Batch of orthogonal matrices
        """
        if self.dim_params > 0:
            # Stack parameters and convert to skew-symmetric matrices
            tensor = convert_to_skewsymm(torch.stack(list(self.params)))
            
            # Compute matrix exponential (equivalent to SO(n) transformation)
            expm_tensor = torch.matrix_exp(tensor)
            return expm_tensor
        else:
            # For 1-dimensional case
            return torch.ones(self.N_batch, 1, 1)

    def loss(self, target_y, predicted_y):
        """
        Compute orthonormality loss between predicted and target matrices.
        
        Args:
            target_y (torch.Tensor): Target matrices
            predicted_y (torch.Tensor): Predicted matrices
        
        Returns:
            torch.Tensor: Mean orthonormality loss
        """
        # Create identity matrices with same batch and size as target
        target_y = torch.tensor(target_y, dtype=torch.float32) if not isinstance(target_y, torch.Tensor) else target_y
        predicted_y = torch.tensor(predicted_y, dtype=torch.float32) if not isinstance(predicted_y, torch.Tensor) else predicted_y
        eye = torch.eye(target_y.shape[2], device=target_y.device).repeat(target_y.shape[0], 1, 1)
        
        # Compute orthonormality loss
        orth_loss = torch.mean(torch.mean(torch.square(
            eye - torch.bmm(predicted_y.transpose(1, 2), target_y)
        ), dim=2), dim=1)
        
        return orth_loss

    def train(self, inputs, target_outputs, learning_rate=0.001, 
              is_using_separate_opt_per_data_point=True):
        """
        Train the Special Orthogonal Groups model.
        
        Args:
            inputs (torch.Tensor or np.ndarray): Input matrices
            target_outputs (torch.Tensor or np.ndarray): Target output matrices
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            is_using_separate_opt_per_data_point (bool, optional): Use separate optimizer 
                                                    for each data point. Defaults to True.
        
        Returns:
            tuple: Losses, mean loss, SO(n) transforms, outputs
        """
        # Ensure inputs and target_outputs are torch tensors
        inputs = torch.tensor(inputs, dtype=torch.float32) if not isinstance(inputs, torch.Tensor) else inputs
        target_outputs = torch.tensor(target_outputs, dtype=torch.float32) if not isinstance(target_outputs, torch.Tensor) else target_outputs
        
        N_batch = inputs.shape[0]
        
        # Setup optimizers
        if is_using_separate_opt_per_data_point and self.dim_params > 0:
            opts = [torch.optim.RMSprop([self.params[i]], lr=learning_rate) 
                    for i in range(N_batch)]
        else:
            # If no parameters or single optimizer, use all parameters
            opt = torch.optim.RMSprop(self.params, lr=learning_rate)
        
        # Compute SO(n) transform and outputs
        SOn_transform = self()
        outputs = torch.bmm(inputs, SOn_transform)
        
        # Compute losses
        raw_losses = self.loss(target_outputs, outputs)
        losses = raw_losses.tolist()
        mean_loss = torch.mean(raw_losses)
        
        # Compute gradients and update parameters
        if self.dim_params > 0:
            if is_using_separate_opt_per_data_point:
                for i in range(N_batch):
                    opts[i].zero_grad()
                    raw_losses[i].backward(retain_graph=True)
                    opts[i].step()
            else:
                opt.zero_grad()
                mean_loss.backward()
                opt.step()
        
        return losses, mean_loss.item(), SOn_transform, outputs


if __name__ == "__main__":
    torch.manual_seed(38)

    N_epoch = 151
    N_batch = 5
    test_num = 1  # 2
    
    n = 3 if test_num == 1 else 5
    dim_params = int(round((n * (n - 1)) / 2))
    
    # Initialize input and ground truth rotation matrices
    input_rot_mat = torch.matrix_exp(convert_to_skewsymm(torch.zeros(N_batch, dim_params)))
    
    if test_num == 1:
        # For 3D case, ground truth is a rotation matrix of π radians around z-axis
        ground_truth_output_params = torch.stack([
            torch.tensor([np.pi if i == 0 else 0 for i in range(dim_params)]) for _ in range(N_batch)
        ])
    else:
        # Random ground truth rotation matrices
        ground_truth_output_params = torch.randn(N_batch, dim_params)
    
    ground_truth_output_rot_mat = torch.matrix_exp(convert_to_skewsymm(ground_truth_output_params))
    print("ground_truth_rot_mat = ", ground_truth_output_rot_mat)

    # Initialize Special Orthogonal Groups
    SOn = SpecialOrthogonalGroups(n=n, N_batch=N_batch, rand_seed=38)

    # Training loop
    SOn_transforms = []
    SOn_transform = SOn()
    for epoch in range(N_epoch):
        SOn_transforms.append(SOn_transform)
        current_losses, current_mean_loss, SOn_transform, _ = SOn.train(
            input_rot_mat, ground_truth_output_rot_mat,
            learning_rate=0.01, is_using_separate_opt_per_data_point=True
        )
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:2d}: ')
            print(f'           mean_loss = {current_mean_loss}')
            print(f'           losses = {current_losses}')
            print(f'           SO{n}_transform = \n{SOn_transforms[-1]}\n')