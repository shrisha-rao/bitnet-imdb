import torch.nn as nn
from bitlinear import BitLinear

def replace_linear_with_bitlinear(model):
    """
    Recursively replace all nn.Linear modules in the model with BitLinear.
    Skips the final classifier head (named 'classifier') if present.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and name != 'classifier':
            # Replace with BitLinear, copying weights and bias
            new_layer = BitLinear(child.in_features, child.out_features, bias=child.bias is not None)
            new_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                new_layer.bias.data = child.bias.data.clone()
            setattr(model, name, new_layer)
        else:
            # Recursively apply to children
            replace_linear_with_bitlinear(child)
