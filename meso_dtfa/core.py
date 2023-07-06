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
    theta: torch.FloatTensor,
    bw: float = 2,
    duration: float = 4,
    sr: float = 2**13,
    delta: int = 0,
):
    """
    Generate an amplitude-modulated chirp signal. 
    The signal is windowed in the time domain by a Gaussian function, whose width
    is controlled by `bw` measured in octaves. All chirps cover a bandwidth `bw`.

    Args:
        theta (Union[torch.FloatTensor]): The carrier frequency (f_c) in Hz, modulator frequency (f_m) in Hz, 
            and chirp rate (gamma) in octaves / second of the chirp signal.
        bw (float, optional): The bandwidth of the chirp signal in octaves. Defaults to 2.
        duration (float, optional): The duration of the chirp signal in seconds. Defaults to 4.
        sr (float, optional): The sample rate of the chirp signal in samples per second. Defaults to 2**13.
        delta (float, optional): applies a time shift in samples
    Returns:
        torch.FloatTensor: The generated amplitude-modulated chirp signal.

    Example:
        >>> theta = [torch.tensor(512.0), torch.tensor(8.0), torch.tensor(1.8)]
        >>> signal = generate_am_chirp(theta, bw=5, duration=2, sr=44100, delta=100)
        >>> signal.shape
    """
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
    y[delta:] = x[:-delta]
    return y


def chirp(t, gamma=0.5, f_c=512):
    chirp_phase = 2 * np.pi * f_c / (gamma * np.log(2)) * (2 ** (gamma * t) - 1)
    return np.sin(chirp_phase)


def am_sine(f_c, f_m, duration=2, sr=2**14):
    t = np.arange(-duration / 2, duration / 2, 1 / sr)
    carrier = np.sin(2 * np.pi * f_c * t)
    modulator = np.sin(2 * np.pi * f_m * t)
    x = carrier * modulator
    return x


def grid2d(x1: float, x2: float, y1: float, y2: float, n: float):
    a = torch.logspace(np.log10(x1), np.log10(x2), n)
    b = torch.logspace(np.log10(y1), np.log10(y2), n)
    X = a.repeat(n)
    Y = b.repeat(n, 1).t().contiguous().view(-1)
    return X, Y


def jtfs_loss(S, x, y):
    loss = torch.norm(S(x) - S(y), p=2)
    return loss


def _ripple(theta, duration, n_partials, sr, window=False):
    """Synthesizes a ripple sound.
    Args:
        theta: [v, w, f0, fm1]
            v (float): octaves per second, w / omega
            omega (float): amount of phase shift at each partial. (Ripple density)
            w (float): Amplitude modulation frequency in Hz. (Ripple drift)
            delta (float): Normalized ripple depth. Value must be in
                the range [0, 1].
            f0 (float): Frequency of the lowest sinusoid in Hz.
            fm1 (float): Frequency of the highest sinusoid in Hz.
        duration (float): Duration of sound in seconds.
        n_partials (int): Number of sinusoids.
        sr (int): Sampling rate in Hz.

    Returns:
        y (torch.tensor): The waveform.
    """
    v, w, f0, fm1 = theta
    device = v.device
    assert len(v.shape) == 2 and v.shape[1] == 1
    phi = 0.0
    # create sinusoids
    m = int(duration * sr)  # total number of samples
    t = torch.linspace(0, duration, int(m)).to(device)[None, None, :]
    i = torch.arange(n_partials).to(device)[None, :]
    # space f0 and highest partial evenly in log domain (divided by # partials)
    f = (f0 * (fm1 / f0) ** (i / (n_partials - 1)))[:, :, None]
    sphi = 0.0  # 2 * torch.pi * torch.rand((1, n_partials, 1))
    s = torch.sin(2 * torch.pi * f * t + sphi)

    # create envelope
    x = torch.log2(f / f0[:, :, None])
    delta = 1.0
    a = 1.0 + delta * torch.sin(
        2 * torch.pi * w[:, :, None] * (t + x / (v[:, :, None])) + phi
    )
    win = torch.hann_window(duration * sr) if window else 1.0
    # create the waveform, summing partials
    y = torch.sum(a * s / torch.sqrt(f), dim=1) * win
    y = y / torch.max(torch.abs(y))

    return y


def ripple(theta, duration, n_partials, sr, window=False):
    """Synthesizes a ripple sound.
    Args:
        theta: [v, w, f0, fm1]
            v (float): octaves per second, w / omega
            omega (float): amount of phase shift at each partial. (Ripple density)
            w (float): Amplitude modulation frequency in Hz. (Ripple drift)
            delta (float): Normalized ripple depth. Value must be in
                the range [0, 1].
            f0 (float): Frequency of the lowest sinusoid in Hz.
            fm1 (float): Frequency of the highest sinusoid in Hz.
        duration (float): Duration of sound in seconds.
        n_partials (int): Number of sinusoids.
        sr (int): Sampling rate in Hz.

    Returns:
        y (torch.tensor): The waveform.
    """
    omega, w, f0, fm1 = theta
    device = omega.device
    phi = 0.0
    # create sinusoids
    m = int(duration * sr)  # total number of samples
    t = torch.linspace(0, duration, int(m)).to(device).reshape((1, -1))
    i = torch.arange(n_partials).to(device).reshape((n_partials, 1))
    # space f0 and highest partial evenly in log domain (divided by # partials)
    f = f0 * (fm1 / f0) ** (i / (n_partials - 1))
    sphi = 0  # 2 * torch.pi * torch.rand((1, n_partials, 1)).to(device)
    s = torch.sin(2 * torch.pi * f * t + sphi)

    # create envelope
    x = torch.log2(f / f0)
    delta = 1.0
    # a = 1.0 + delta * torch.sin(
    #     2 * torch.pi * w[:, :, None] * (t + x / (v[:, :, None])) + phi
    # )
    a = 1.0 + delta * torch.sin(2 * torch.pi * (w * t + x * omega) + phi)
    # win = torch.hann_window(duration * sr) if window else 1.0
    # create the waveform, summing partials
    y = torch.sum(a * s, dim=0)  # / torch.sqrt(f)
    y = y / torch.sum(torch.abs(y))

    return y
