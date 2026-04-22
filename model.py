import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Store for reference (good practice)
        self.in_features = in_features
        self.out_features = out_features

        # 1. Standard weight and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # 2. Gate scores (same shape as weight)
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        # Convert gate_scores → gates in [0, 1]
        gates = torch.sigmoid(self.gate_scores)

        # Apply pruning
        pruned_weights = self.weight * gates

        # Linear operation
        output = F.linear(x, pruned_weights, self.bias)

        return output


# Test block 
if __name__ == "__main__":
    layer = PrunableLinear(4, 2)

    x = torch.randn(1, 4)
    output = layer(x)

    print("Input:", x)
    print("Output:", output)

    # Check gates
    gates = torch.sigmoid(layer.gate_scores)
    print("Gate values shape:", gates.shape)
    print("Sample gates:", gates[0][:5])