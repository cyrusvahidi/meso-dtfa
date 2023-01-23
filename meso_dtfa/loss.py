import torch, torch.nn as nn
from kymatio.torch import TimeFrequencyScattering

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
            return torch.norm(x - y, p=self.p)

    def forward(self, x, y, transform_y=True):
        loss = torch.tensor(0.0).type_as(x)
        for op in self.ops:
            loss += self.dist(op(x), op(y) if transform_y else y)
        loss /= len(self.ops)
        return loss


class TimeFrequencyScatteringLoss(DistanceLoss):

    def __init__(
        self, shape, Q=(8, 2), J=12, J_fr=3, Q_fr=2, F=None, T=None, format="time", p=2.0
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
            format=self.format
        ).cuda()
        self.ops = [S]


class MultiScaleSpectralLoss(DistanceLoss):

    def __init__(
        self, max_n_fft=2048, num_scales=6, hop_lengths=None, mag_w=1.0, logmag_w=0.0, p=1.0
    ):
        super().__init__(p=p)
        assert max_n_fft // 2**(num_scales - 1) > 1
        self.max_n_fft = 2048
        self.n_ffts = [max_n_fft // (2 ** i) for i in range(num_scales)]
        self.hop_lengths = [n // 4 for n in self.n_ffts] if not hop_lengths else hop_lengths
        self.mag_w = mag_w 
        self.logmag_w = logmag_w

        self.create_ops()

    def create_ops(self):
        self.ops = [MagnitudeSTFT(n_fft, self.hop_lengths[i]) for i, n_fft in enumerate(self.n_ffts)]

    
class MagnitudeSTFT(nn.Module):

    def __init__(self, n_fft, hop_length):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length 

    def forward(self, x):
        return torch.stft(x, 
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          window=torch.hann_window(self.n_fft).type_as(x),
                          return_complex=True).abs()