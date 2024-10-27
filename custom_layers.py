import torch
import torch.nn as nn
import torch.nn.functional as F

import torch


class AdaptiveAvgPool2d:
    def __init__(self, output_size):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def forward(self, x):
        batch_size, channels, H, W = x.shape
        output_H, output_W = self.output_size

        stride_H = H // output_H
        stride_W = W // output_W
        kernel_H = H - (output_H - 1) * stride_H
        kernel_W = W - (output_W - 1) * stride_W

        pooled = torch.zeros(
            (batch_size, channels, output_H, output_W), device=x.device)

        for i in range(output_H):
            for j in range(output_W):
                pooled[:, :, i, j] = x[:, :, i * stride_H: i * stride_H + kernel_H,
                                       j * stride_W: j * stride_W + kernel_W].mean(dim=(2, 3))

        return pooled

    def __call__(self, x):
        return self.forward(x)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(
            kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(
            padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(
            dilation, tuple) else (dilation, dilation)
        self.use_bias = bias  # Store whether bias should be used or not

        # Weights as learnable parameters
        self.weights = nn.Parameter(torch.randn(
            out_channels, in_channels, *self.kernel_size))

        # Initialize bias only if bias is set to True
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)  # No bias parameter

    def forward(self, x):
        # Input shape: (batch_size, in_channels, height, width)
        batch_size, _, height, width = x.shape

        # Apply padding using F.pad
        x = F.pad(x, (self.padding[1], self.padding[1],
                  self.padding[0], self.padding[0]))

        # Unfold the input into patches (optimized for performance)
        unfolded_x = F.unfold(x, kernel_size=self.kernel_size,
                              dilation=self.dilation, stride=self.stride)

        # Shape the unfolded input to be batch_size x (in_channels * kernel_height * kernel_width) x (output_height * output_width)
        # Weights shape: out_channels x (in_channels * kernel_height * kernel_width)
        unfolded_x = unfolded_x.view(
            batch_size, self.in_channels * self.kernel_size[0] * self.kernel_size[1], -1)

        # Reshape weights for efficient matrix multiplication
        weights_flat = self.weights.view(self.out_channels, -1)

        # Perform matrix multiplication: weights_flat (out_channels, in_channels * kernel_size * kernel_size)
        # and unfolded_x (batch_size, in_channels * kernel_size * kernel_size, num_patches)
        conv_out = torch.einsum('oi,bip->bop', weights_flat, unfolded_x)

        # Add bias if it exists
        if self.use_bias:
            conv_out = conv_out + self.bias.view(1, -1, 1)

        # Compute output height and width
        output_height = (height + 2 * self.padding[0] - self.dilation[0] * (
            self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        output_width = (width + 2 * self.padding[1] - self.dilation[1] * (
            self.kernel_size[1] - 1) - 1) // self.stride[1] + 1

        # Reshape to (batch_size, out_channels, height_out, width_out)
        return conv_out.view(batch_size, self.out_channels, output_height, output_width)


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2d, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # Learnable parameters (gamma and beta) if affine is True
        if self.affine:
            self.weight = nn.Parameter(torch.ones(
                num_features))  # gamma, scale factor
            self.bias = nn.Parameter(torch.zeros(
                num_features))   # beta, shift factor
        else:
            self.weight = None
            self.bias = None

        # Running stats (mean and variance) if track_running_stats is True
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.running_mean = None
            self.running_var = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Ensure input is in the format [batch_size, num_channels, height, width]
        assert x.dim() == 4, "Input should be a 4D tensor (batch_size, num_channels, height, width)"

        # Compute mean and variance along batch, height, and width dimensions
        if self.training and self.track_running_stats:
            # Calculate batch mean and variance across batch, height, and width
            # Mean over the batch, height, and width dimensions
            batch_mean = x.mean(dim=[0, 2, 3])
            # Variance over the batch, height, and width
            batch_var = x.var(dim=[0, 2, 3], unbiased=False)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * \
                self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * \
                self.running_var + self.momentum * batch_var
        else:
            # Use running mean and variance during evaluation
            batch_mean = self.running_mean
            batch_var = self.running_var

        # Normalize the input
        x_normalized = (x - batch_mean[None, :, None, None]) / \
            (batch_var[None, :, None, None] + self.eps).sqrt()

        # Apply learnable affine transformation if enabled
        if self.affine:
            x_normalized = x_normalized * \
                self.weight[None, :, None, None] + \
                self.bias[None, :, None, None]

        return x_normalized


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Apply padding to the input tensor
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding,
                      self.padding), mode='constant', value=float('-inf'))

        # Get input dimensions
        n, c, h, w = x.size()

        # Calculate output dimensions
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1

        # Initialize output tensor
        out = torch.zeros((n, c, out_h, out_w), device=x.device)

        # Perform max pooling
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                # Apply max over the height, then over the width
                max_over_height = torch.max(
                    x[:, :, h_start:h_end, w_start:w_end], dim=-2).values
                out[:, :, i, j] = torch.max(max_over_height, dim=-1).values

        return out


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        # Manual ReLU operation
        return torch.maximum(x, torch.zeros_like(x))
