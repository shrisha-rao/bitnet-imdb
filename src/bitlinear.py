import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear(nn.Linear):
    """
    BitLinear layer with ternary weights {-1, 0, 1} and straight-through estimator.
    Based on the BitNet 1.58bit paper: weights are scaled per layer.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        # Store a full-precision copy of weights (inherited from nn.Linear)

    def quantize_weights(self):
        """
        Quantize weights to ternary values: {-alpha, 0, alpha}
        where alpha is the mean absolute value of the weights.
        """
        w = self.weight  # full precision weights

        # Compute per-layer scale (alpha) as mean absolute value
        alpha = w.abs().mean().clamp(min=1e-8)

        # Ternary quantization: values > 0.5*alpha become +alpha, < -0.5*alpha become -alpha, else 0
        ternary = torch.where(w > 0.5 * alpha, alpha, torch.where(w < -0.5 * alpha, -alpha, 0.0))

        return ternary

    def forward(self, x):
        # Quantize weights for forward pass
        quantized_w = self.quantize_weights()

        # Use straight-through estimator: forward uses quantized weights,
        # backward passes gradients as if we used full-precision weights.
        # This is achieved by detaching quantized weights and adding the gradient of full-precision weights.
        w_ste = self.weight + (quantized_w - self.weight).detach()
        return F.linear(x, w_ste, self.bias)
