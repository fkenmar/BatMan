from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_channels))
            self.bias_imag = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        real = self.real(z.real) - self.imag(z.imag)
        imag = self.real(z.imag) + self.imag(z.real)
        if self.bias_real is not None and self.bias_imag is not None:
            real = real + self.bias_real.view(1, -1, 1)
            imag = imag + self.bias_imag.view(1, -1, 1)
        return torch.complex(real, imag)


class ComplexBatchNorm1d(nn.Module):
    """Lightweight complex BN using shared real-valued BN modules per component."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.real = nn.BatchNorm1d(channels)
        self.imag = nn.BatchNorm1d(channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.real(z.real), self.imag(z.imag))


class ComplexModReLU(nn.Module):
    """Phase-preserving activation: changes magnitude while keeping angle."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(z)
        gate = F.relu(magnitude + self.bias.view(1, -1, 1))
        return z * (gate / (magnitude + 1e-6))


class ComplexDropout(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return z
        keep_prob = 1.0 - self.p
        mask = torch.empty_like(z.real).bernoulli_(keep_prob).div_(keep_prob)
        return z * mask


class ComplexSpectralBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.05) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            ComplexBatchNorm1d(channels),
            ComplexModReLU(channels),
            ComplexConv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            ComplexDropout(dropout),
            ComplexBatchNorm1d(channels),
            ComplexModReLU(channels),
            ComplexConv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


class FourierComplexAMC(nn.Module):
    """Fourier-native, phase-preserving AMC classifier.

    Input shape is ``(batch, time, 2)`` with I and Q in the last dimension.
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
        self.blocks = nn.Sequential(
            *[
                ComplexSpectralBlock(
                    width,
                    kernel_size=kernel_size,
                    dilation=dilations[i % len(dilations)],
                    dropout=dropout,
                )
                for i in range(depth)
            ]
        )
        self.norm = ComplexBatchNorm1d(width)
        self.act = ComplexModReLU(width)
        self.classifier = nn.Sequential(
            nn.Linear(width * 4, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, num_classes),
        )

    def _to_spectrum(self, iq: torch.Tensor) -> torch.Tensor:
        if iq.ndim != 3 or iq.shape[-1] != 2:
            raise ValueError(f"Expected input shape (batch, time, 2), got {tuple(iq.shape)}")
        z = torch.complex(iq[..., 0], iq[..., 1])
        spectrum = torch.fft.fft(z, dim=1, norm="ortho")
        spectrum = torch.fft.fftshift(spectrum, dim=1)
        if self.keep_bins is not None and self.keep_bins < spectrum.shape[1]:
            center = spectrum.shape[1] // 2
            half = self.keep_bins // 2
            start = center - half
            end = start + self.keep_bins
            spectrum = spectrum[:, start:end]
        return spectrum.unsqueeze(1)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        z = self._to_spectrum(iq)
        z = self.stem(z)
        z = self.blocks(z)
        z = self.act(self.norm(z))

        magnitude = torch.abs(z)
        avg_mag = F.adaptive_avg_pool1d(magnitude, 1).squeeze(-1)
        max_mag = F.adaptive_max_pool1d(magnitude, 1).squeeze(-1)
        mean_real = F.adaptive_avg_pool1d(z.real, 1).squeeze(-1)
        mean_imag = F.adaptive_avg_pool1d(z.imag, 1).squeeze(-1)
        features = torch.cat([avg_mag, max_mag, mean_real, mean_imag], dim=1)
        return self.classifier(features)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
