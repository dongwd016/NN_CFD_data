import numpy as np
import torch
import torch.nn as nn


class DNN(nn.Module):
    """
    input_dim: input dimension of a neural network model, (T, P, Y, dt) at time t
    hidden_dims: a list of hidden dimensions
    output_dim: output dimension of a neural network model, (T, P, Y) at time t+dt
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim=1,
        activation_function=nn.ReLU(),
    ):
        super().__init__()
        layer_dim_lists = [input_dim] + hidden_dims + [output_dim]
        layers = [nn.Linear(layer_dim_lists[i - 1], layer_dim_lists[i]) for i in range(1, len(layer_dim_lists))]
        self.linear_layers = nn.ModuleList(layers)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.linear_layers[0](x)
        x = self.activation_function(x)
        for layer in self.linear_layers[1:-1]:
            x = layer(x)
            x = self.activation_function(x)
        x = self.linear_layers[-1](x)
        return x


input_dim = 12
output_dim = 11
nn_input_dict = {
    "input_dim": input_dim,
    "hidden_dims": [1024, 10, 1024],
    "output_dim": output_dim,
    "activation_function": nn.ReLU(),
}
model = DNN(**nn_input_dict)
model.load_state_dict(torch.load("model_state_dict.pth"))
print(model)
