from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class ComplexConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.real = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.imag = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bias_real = mx.zeros((out_channels,)) if bias else None
        self.bias_imag = mx.zeros((out_channels,)) if bias else None

    def __call__(self, z: mx.array) -> mx.array:
        real = self.real(z.real) - self.imag(z.imag)
        imag = self.real(z.imag) + self.imag(z.real)
        if self.bias_real is not None and self.bias_imag is not None:
            real = real + self.bias_real.reshape(1, 1, -1)
            imag = imag + self.bias_imag.reshape(1, 1, -1)
        return real + 1j * imag


class ComplexRMSNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((channels,))
        self.eps = eps

    def __call__(self, z: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(mx.square(mx.abs(z)), axis=-1, keepdims=True) + self.eps)
        return z / rms * self.weight.reshape(1, 1, -1)


class ComplexModReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bias = mx.zeros((channels,))

    def __call__(self, z: mx.array) -> mx.array:
        magnitude = mx.abs(z)
        gate = mx.maximum(magnitude + self.bias.reshape(1, 1, -1), 0.0)
        return z * (gate / (magnitude + 1e-6))


class ComplexSpectralBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.norm1 = ComplexRMSNorm(channels)
        self.act1 = ComplexModReLU(channels)
        self.conv1 = ComplexConv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = ComplexRMSNorm(channels)
        self.act2 = ComplexModReLU(channels)
        self.conv2 = ComplexConv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)

    def __call__(self, z: mx.array) -> mx.array:
        residual = z
        z = self.conv1(self.act1(self.norm1(z)))
        z = self.conv2(self.act2(self.norm2(z)))
        return residual + z


class FourierComplexAMCMLX(nn.Module):
    """MLX/Metal Fourier-native complex AMC classifier.

    Input shape is ``(batch, time, 2)`` with I and Q in the last dimension.
    MLX Conv1d expects channels-last tensors, so spectral features are shaped
    as ``(batch, frequency, channels)`` throughout the complex backbone.
    """

    def __init__(
        self,
        num_classes: int = 24,
        width: int = 64,
        depth: int = 6,
        kernel_size: int = 5,
        dropout: float = 0.1,
        keep_bins: int | None = None,
    ) -> None:
        super().__init__()
        self.keep_bins = keep_bins
        self.stem = ComplexConv1d(1, width, kernel_size=7, padding=3)
        dilations = [1, 2, 4, 8]
        self.blocks = [
            ComplexSpectralBlock(
                width,
                kernel_size=kernel_size,
                dilation=dilations[i % len(dilations)],
            )
            for i in range(depth)
        ]
        self.norm = ComplexRMSNorm(width)
        self.act = ComplexModReLU(width)
        self.fc1 = nn.Linear(width * 4, width * 2)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(width * 2, num_classes)

    def _to_spectrum(self, iq: mx.array) -> mx.array:
        if len(iq.shape) != 3 or iq.shape[-1] != 2:
            raise ValueError(f"Expected input shape (batch, time, 2), got {iq.shape}")
        z = iq[..., 0] + 1j * iq[..., 1]
        spectrum = mx.fft.fft(z, axis=1) / math.sqrt(iq.shape[1])
        spectrum = mx.fft.fftshift(spectrum, axes=1)
        if self.keep_bins is not None and self.keep_bins < spectrum.shape[1]:
            center = spectrum.shape[1] // 2
            half = self.keep_bins // 2
            start = center - half
            spectrum = spectrum[:, start : start + self.keep_bins]
        return spectrum[..., None]

    def __call__(self, iq: mx.array) -> mx.array:
        z = self.stem(self._to_spectrum(iq))
        for block in self.blocks:
            z = block(z)
        z = self.act(self.norm(z))

        magnitude = mx.abs(z)
        avg_mag = mx.mean(magnitude, axis=1)
        max_mag = mx.max(magnitude, axis=1)
        mean_real = mx.mean(z.real, axis=1)
        mean_imag = mx.mean(z.imag, axis=1)
        features = mx.concatenate([avg_mag, max_mag, mean_real, mean_imag], axis=1)
        features = self.dropout(nn.gelu(self.fc1(features)))
        return self.fc2(features)
