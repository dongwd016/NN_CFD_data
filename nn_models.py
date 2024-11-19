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
        dropout_prob=0.0,
        activation_function=nn.ReLU(),
        torch_init=False,
        torch_init_gain=1.0,
        output_activation_function=None,
        dropout_location="first",
    ):
        super().__init__()
        layer_dim_lists = [input_dim] + hidden_dims + [output_dim]
        layers = [nn.Linear(layer_dim_lists[i - 1], layer_dim_lists[i]) for i in range(1, len(layer_dim_lists))]
        self.linear_layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.activation_function = activation_function
        self.output_activation_function = output_activation_function
        if torch_init:
            for layer in self.linear_layers:
                nn.init.xavier_uniform_(layer.weight, gain=torch_init_gain)
        self.dropout_location = dropout_location

    def forward(self, x):
        x = self.linear_layers[0](x)
        x = self.activation_function(x)
        if self.dropout_location == "first":
            x = self.dropout(x)
        for layer in self.linear_layers[1:-1]:
            x = layer(x)
            x = self.activation_function(x)
        if self.dropout_location == "last":
            x = self.dropout(x)
        x = self.linear_layers[-1](x)
        if self.output_activation_function is not None:
            x = self.output_activation_function(x)
        return x


input_dim = 12
output_dim = 11
nn_input_dict = {
    "input_dim": input_dim,
    "hidden_dims": [1024, 10, 1024],
    "output_dim": output_dim,
    "dropout_prob": 0.0,
    "activation_function": nn.ReLU(),
    "torch_init": False,
    "torch_init_gain": np.sqrt(2),
}
model = DNN(**nn_input_dict)
model.load_state_dict(torch.load("model_state_dict.pth"))
print(model)
