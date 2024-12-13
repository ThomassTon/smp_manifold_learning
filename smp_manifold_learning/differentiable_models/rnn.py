import torch
from torch.nn import Module, RNN, Linear, Dropout, Tanh, ReLU, BatchNorm1d
import smp_manifold_learning.differentiable_models.nn as nn
import smp_manifold_learning.differentiable_models.utils as utils


class EqualityConstraintManifoldRNN(Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, activation='tanh',
                 use_batch_norm=False, drop_p=0.0, name='', is_training=False, device='cpu'):
        super().__init__()
        self.name = name
        self.is_training = is_training
        self.device = device
        self.dim_ambient = input_dim
        self.N_constraints = output_dim

        # RNN layers
        self.rnn = torch.nn.RNN(input_dim, hidden_sizes[0], len(hidden_sizes), batch_first=True, nonlinearity=activation)
        self.final_linear = Linear(hidden_sizes[-1], output_dim)

        self.activation = Tanh() if activation == 'tanh' else ReLU()
        self.dropout = Dropout(drop_p) if drop_p > 0.0 else None
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norms = torch.nn.ModuleList([BatchNorm1d(size) for size in hidden_sizes])
        
        self.to(self.device)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def train(self):
        super().train()
        self.is_training = True

    def eval(self):
        super().eval()
        self.is_training = False

    def forward(self, x):
        # Assuming x has shape (batch_size, seq_len, input_dim)
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Take the last hidden state
        if self.use_batch_norm:
            x = self.batch_norms[-1](x)
        if self.dropout:
            x = self.dropout(x)
        x = self.final_linear(x)
        return x

    def y_torch(self, x):
        return self(x)

    def y(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            y_torch = self.y_torch(x)
        return y_torch.cpu().numpy()

    def get_loss_components(self, data_dict):
        loss_components = dict()
        
        data = data_dict['sequential_data'].to(self.device)
        norm_level_data_torch = utils.convert_into_at_least_2d_pytorch_tensor(data_dict['norm_level_data']).to(self.device)
        norm_level_weight_torch = utils.convert_into_at_least_2d_pytorch_tensor(data_dict['norm_level_weight']).to(self.device)

        y_torch = self.y_torch(data)

        # (level set) prediction error
        norm_level_wnmse_per_dim = utils.compute_wnmse_per_dim(
            prediction=torch.norm(y_torch, dim=1).unsqueeze(1),
            ground_truth=norm_level_data_torch,
            weight=norm_level_weight_torch
        )

        loss_components['norm_level_wnmse_per_dim'] = norm_level_wnmse_per_dim
        return loss_components

    def print_inference_result(self, data_dict, prefix_name=''):
        loss_components = self.get_loss_components(data_dict)
        np_loss_components = {key: loss_components[key].cpu().detach().numpy() for key in loss_components}
        print(prefix_name)
        for key in loss_components:
            print(f"   {prefix_name}_{key} = {np_loss_components[key]}")
        return np_loss_components

    def print_prediction_stats(self, data_dict, axis=None):
        pred = self.y(data_dict['sequential_data'])

        # mean and std of prediction
        pred_mean = pred.mean(axis=axis)
        pred_std = pred.std(axis=axis)

        print(f"Prediction Stats: [mean, std] = [{pred_mean}, {pred_std}]")
        return None
