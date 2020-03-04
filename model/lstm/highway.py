import torch.nn as nn
import torch
# https://github.com/allenai/allennlp/blob/master/allennlp/modules/highway.py
class Highway(nn.Module):
    def __init__(self, input_dim, num_layers=1):
        super(Highway, self).__init__()

        self._layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        for layer in self._layers:
            # Bias the highway layer to just carry its input forward.
            # Set the bias on B(x) to be positive, then g will be biased to be high
            # The bias on B(x) is the second half of the bias vector in each linear layer.
            print(layer.bias,layer.bias.shape)
            layer.bias[input_dim:].data.fill_(1)
            print(layer.bias,layer.bias.shape)

    def forward(self, inputs):
        current_inputs = inputs
        for layer in self._layers:
            linear_part = current_inputs
            projected_inputs = layer(current_inputs)

            nonlinear_part, gate = projected_inputs.chunk(2, dim=-1)
            nonlinear_part = nn.functional.relu(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_inputs = gate * linear_part + (1 - gate) * nonlinear_part
        return current_inputs

# h = Highway(30,2)
# x = torch.rand(10,25,30)
# print(x.shape)
# y = h(x)
# print(y.shape)