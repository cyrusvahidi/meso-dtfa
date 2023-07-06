import torch, torch.nn as nn, numpy as np
from kymatio.torch import TimeFrequencyScattering


def cosine_distance(x, y):
    """Compute the cosine distance between two 2D tensors.""" 
    sim = torch.nn.functional.cosine_similarity(x, y, dim=0)
    dist = 1 - sim
    dist /= 2 # Normalize to range [0,1]
    dist = dist.mean()
    return dist


class DistanceLoss(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p

    def create_ops(*args, **kwargs):
        raise NotImplementedError

    def dist(self, x, y):
        if self.p == 1.0:
            return torch.abs(x - y).mean()
        elif self.p == 2.0:
            return torch.norm(x - y, p=2.0)
        elif self.p == "cosine":
            return cosine_distance(x, y)

    def forward(self, x, y, transform_y=True):
        loss = torch.tensor(0.0).type_as(x)
        for op in self.ops:
            loss += self.dist(op(x), op(y) if transform_y else y)
        loss /= len(self.ops)
        return loss


class TimeFrequencyScatteringLoss(DistanceLoss):
    def __init__(
        self,
        shape,
        Q=(8, 2),
        J=12,
        J_fr=3,
        Q_fr=2,
        F=None,
        T=None,
        format="time",
        p=2.0,
    ):
        super().__init__(p=p)

        self.shape = shape
        self.Q = Q
        self.J = J
        self.J_fr = J_fr
        self.F = F
        self.Q_fr = Q_fr
        self.T = T
        self.format = format
        self.create_ops()

    def create_ops(self):
        S = TimeFrequencyScattering(
            shape=self.shape,
            Q=self.Q,
            J=self.J,
            J_fr=self.J_fr,
            Q_fr=self.Q_fr,
            T=self.T,
            F=self.F,
            format=self.format,
        ).cuda()
        self.ops = [S]

class MultiScaleSpectralLoss(DistanceLoss):
    """
    Multi-scale spectral loss module.

    Args:
        max_n_fft (int, optional): The maximum size of the FFT (Fast Fourier Transform). Defaults to 2048.
        num_scales (int, optional): The number of scales to consider. Defaults to 6.
        hop_lengths (list, optional): The hop lengths for each scale. If not provided, they are computed automatically. Defaults to None.
        mag_w (float, optional): The weight for the magnitude component. Defaults to 1.0.
        logmag_w (float, optional): The weight for the log-magnitude component. Defaults to 0.0.
        p (float, optional): The exponent value for the distance metric. Defaults to 1.0.

    Notes:
        - The `max_n_fft` parameter should be divisible by 2 raised to the power of (`num_scales` - 1).
        - If `hop_lengths` are not provided, they are automatically computed based on the `n_ffts` of each scale.

    Example:
        >>> loss = MultiScaleSpectralLoss(max_n_fft=4096, num_scales=4, mag_w=0.8, logmag_w=0.2, p=2.0)
    """
    def __init__(
        self,
        max_n_fft=2048,
        num_scales=6,
        hop_lengths=None,
        mag_w=1.0,
        logmag_w=0.0,
        p=1.0,
    ):
        super().__init__(p=p)
        assert max_n_fft // 2 ** (num_scales - 1) > 1
        self.max_n_fft = 2048
        self.n_ffts = [max_n_fft // (2**i) for i in range(num_scales)]
        self.hop_lengths = (
            [n // 4 for n in self.n_ffts] if not hop_lengths else hop_lengths
        )
        self.mag_w = mag_w
        self.logmag_w = logmag_w

        self.create_ops()

    def create_ops(self):
        self.ops = [
            MagnitudeSTFT(n_fft, self.hop_lengths[i])
            for i, n_fft in enumerate(self.n_ffts)
        ]


class MagnitudeSTFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).type_as(x),
            return_complex=True,
        ).abs()


class TimeFrequencyScatteringS2Loss(DistanceLoss):
    """
     Time-Frequency Scattering transform loss

    Args:
        shape (tuple): The shape of the input signals in the form (num_samples, ).
        Q (tuple, optional): The quality factors for the filters at each order. Defaults to (8, 2).
        J (int, optional): The number of scales for the scattering transform. Defaults to 12.
        J_fr (int, optional): The number of scales for the frequency scattering filterbank. Defaults to 3.
        Q_fr (int, optional): The quality factor for the filters in the frequential filterbank. Defaults to 2.
        F (int, optional): Frequency averaging scale. Defgault to 2**J_fr
        T (int, optional): Temporal averaging scale. Defaults to 2**J
        format (str, optional): output format with options ["time", "joint"]
        p (float, optional): The exponent value for the distance metric. Defaults to 2.0.

    Example:
        >>> loss = TimeFrequencyScatteringS2Loss(shape=(batch_size, channels, height, width), Q=(6, 3), J=10, J_fr=2, Q_fr=3)
    """
    def __init__(
        self,
        shape,
        Q=(8, 2),
        J=12,
        J_fr=3,
        Q_fr=2,
        F=None,
        T=None,
        format="time",
        p=2.0,
    ):
        super().__init__(p=p)

        self.shape = shape
        self.Q = Q
        self.J = J
        self.J_fr = J_fr
        self.F = F
        self.Q_fr = Q_fr
        self.T = T
        self.format = format
        self.create_ops()

    def create_ops(self):
        S = TimeFrequencyScattering(
            shape=self.shape,
            Q=self.Q,
            J=self.J,
            J_fr=self.J_fr,
            Q_fr=self.Q_fr,
            T=self.T,
            F=self.F,
            format=self.format,
        ).cuda()
        self.ops = [S]
        self.idxs = np.where(S.meta()["order"] == 2)

    def forward(self, x, y, transform_y=True):
        """
        Compute the forward pass of the loss.

        Args:
            x (torch.Tensor): The input signal.
            y (torch.Tensor): The target signal.
            transform_y (bool, optional): Whether to apply the transform on the target signal. Defaults to True.

        Returns:
            torch.Tensor: The computed loss value.
        """
        loss = torch.tensor(0.0).type_as(x)
        for op in self.ops:
            loss += self.dist(
                op(x)[0, self.idxs[0][0]:], op(y)[0, self.idxs[0][0]:] if transform_y else y
            )
        loss /= len(self.ops)
        return loss


class WeightedTimeFrequencyScatteringLoss(DistanceLoss):
    def __init__(
        self,
        shape,
        Q=(8, 2),
        J=12,
        J_fr=3,
        Q_fr=2,
        F=None,
        T=None,
        format="time",
        p=2.0,
        weights=(1.0, 1.0)
    ):
        super().__init__(p=p)

        self.shape = shape
        self.Q = Q
        self.J = J
        self.J_fr = J_fr
        self.F = F
        self.Q_fr = Q_fr
        self.T = T
        self.format = format
        self.create_ops()
        self.weights = weights

    def create_ops(self):
        S = TimeFrequencyScattering(
            shape=self.shape,
            Q=self.Q,
            J=self.J,
            J_fr=self.J_fr,
            Q_fr=self.Q_fr,
            T=self.T,
            F=self.F,
            format=self.format,
        ).cuda()
        self.ops = [S]
        self.idxs = (np.where(S.meta()["order"] == 1), np.where(S.meta()["order"] == 2))

    def forward(self, x, y, transform_y=True):
        loss = torch.tensor(0.0).type_as(x)
        for op in self.ops:
            Sx = op(x)
            Sy = op(y)[0] if transform_y else y
            for i, w in enumerate(self.weights):
                loss += w * self.dist(
                    Sx[self.idxs[i]], Sy[self.idxs[i]]
                )
        loss /= len(self.ops)
        return loss