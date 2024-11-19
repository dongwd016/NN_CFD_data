# NN_CFD_data

This repository contains the data used for the neural network reduced model for CFD simulation project.

The neural network takes current state (temperature, pressure, species mass fraction) and a desired time step within 10<sup>-10</sup> s to 10 <sup>-8</sup> s as input and predicts the next state of the system (temperature, pressure, species mass fraction) within the next time step. The neural network architecture is a fully connected DNN with size: 12 * 1024 * 10 * 1024 * 11. The activation function is ReLU (max(x,0)). It is applied to all layers except for the output. Weight and bias matrices are stored in both *.txt and *.pth format. For the usage of *.pth state_dict file, please refer to the python code in "nn_models.py" file.

When using the neural network, you need to scale the input data with the provided mean and std files for input before feeding it to the neural network. The nerual network output also needs to be scaled back to the real physical values using the provided mean and std files for output.

- NN input = (T(t),P(t),Y(t),log10(dt) - mean_input) / std_input
- T(t+dt),P(t+dt),Y(t+dt) = NN_output * std_output + mean_output

## h5 file

There is an h5 file ("model_state_dict.h5") containing both the state_dict of the neural network and the scaling information of the input and output of the neural network. It's structure is as follows:

- linear_layers
  - 0
    - weight
    - bias
  - 1
    - weight
    - bias
  - 2
    - weight
    - bias
  - 3
    - weight
    - bias
- mean_input
- std_input
- mean_output
- std_output

## txt files shape

- linear_layers.0.weight: 1024 * 12
- linear_layers.0.bias:   1024 * 1
- linear_layers.1.weight: 10 * 1024
- linear_layers.1.bias:   10 * 1
- linear_layers.2.weight: 1024 * 10
- linear_layers.2.bias:   1024 * 1
- linear_layers.3.weight: 11 * 1024
- linear_layers.3.bias:   11 * 1

<br/>

- mean_input: 12 * 1
- std_input:  12 * 1
- mean_output: 11 * 1
- std_output:  11 * 1

## pth file structure after loading

```
DNN(
  (linear_layers): ModuleList(
    (0): Linear(in_features=12, out_features=1024, bias=True)
    (1): Linear(in_features=1024, out_features=10, bias=True)
    (2): Linear(in_features=10, out_features=1024, bias=True)
    (3): Linear(in_features=1024, out_features=11, bias=True)
  )
  (dropout): Dropout(p=0.0, inplace=False)
  (activation_function): ReLU()
)
```
