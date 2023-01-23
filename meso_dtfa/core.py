import torch, numpy as np


def gauss_window(M: float, std: torch.FloatTensor, sym: bool = True):
    """Gaussian window converted from scipy.signal.gaussian"""
    if M < 1:
        return torch.array([])
    if M == 1:
        return torch.ones(1, "d").type_as(std)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    n = n.type_as(std)

    sig2 = 2 * std * std
    w = torch.exp(-(n**2) / sig2)
    if not sym and not odd:
        w = w[:-1]
    return w


def generate_am_chirp(
    theta: torch.FloatTensor, bw: float = 2, duration: float = 4, sr: float = 2**13, delta=0
):
    f_c, f_m, gamma = theta[0], theta[1], theta[2]
    t = torch.arange(-duration / 2, duration / 2, 1 / sr).type_as(f_m)
    carrier = sine(f_c / (gamma * np.log(2)) * (2 ** (gamma * t) - 1))
    modulator = sine(t * f_m)
    sigma0 = 0.1
    window_std = (torch.tensor(sigma0 * bw).type_as(gamma)) / gamma
    window = gauss_window(duration * sr, std=window_std * sr)
    x = carrier * (modulator if f_m != 0 else 1.0) * window * float(gamma)
    if delta:
        x = time_shift(x, delta)
    return x


def sine(phi):
    return torch.sin(2 * torch.pi * phi)


def time_shift(x, delta):
    y = torch.zeros_like(x)
    y[delta:] = x[:- delta]
    return y


def chirp(t, gamma=0.5, f_c=512):
    chirp_phase = 2*np.pi*f_c / (gamma*np.log(2)) * (2 ** (gamma*t) - 1)
    return np.sin(chirp_phase)


def am_sine(f_c, f_m, duration=2, sr=2**14):
    t = np.arange(-duration/2, duration/2, 1/sr)
    carrier = np.sin(2*np.pi*f_c * t)
    modulator = np.sin(2 * np.pi * f_m * t)
    x = carrier * modulator
    return x


def grid2d(
    x1: float, x2: float, y1: float, y2: float, n: float
):
    a = torch.logspace(np.log10(x1), np.log10(x2), n)
    b = torch.logspace(np.log10(y1), np.log10(y2), n)
    X = a.repeat(n)
    Y = b.repeat(n, 1).t().contiguous().view(-1)
    return X, Y


def jtfs_loss(S, x, y):
    loss = torch.norm(S(x) - S(y), p=2)
    return loss