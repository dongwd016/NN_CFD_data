# NN_CFD_data

This repository contains the data used for the neural network reduced model for CFD simulation project.

The neural network architecture is a fully connected DNN with size: 12 * 1024 * 10 * 1024 * 11.

Weight and bias matrix stored in both *.txt and *.pth format. For the usage of *.pth state_dict file, please refer to the python code in "nn_models.py" file.

When using the neural network, you need to scale the input data with the provided mean and std files for input before feeding it to the neural network. The nerual network output also needs to be scaled back to the real physical values using the provided mean and std files for output.

NN input = (T(t),P(t),Y(t),log10(dt) - mean_input) / std_input

T(t+dt),P(t+dt),Y(t+dt) = NN_output * std_output + mean_output

## txt files shape

- linear_layers.0.weight: 1024 * 12
- linear_layers.0.bias:   1024 * 1
- linear_layers.1.weight: 10 * 1024
- linear_layers.1.bias:   10 * 1
- linear_layers.2.weight: 1024 * 10
- linear_layers.2.bias:   1024 * 1
- linear_layers.3.weight: 11 * 1024
- linear_layers.3.bias:   11 * 1

- mean_input: 12 * 1
- std_input:  12 * 1
- mean_output: 11 * 1
- std_output:  11 * 1
