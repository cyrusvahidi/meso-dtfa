import numpy as np, torch, random, fire
from tqdm import tqdm

from meso_dtfa.core import generate_am_chirp, grid2d, ripple
from meso_dtfa.loss import TimeFrequencyScatteringLoss, MultiScaleSpectralLoss, WeightedTimeFrequencyScatteringLoss
from meso_dtfa.plot import plot_contour_gradient, mesh_plot_3d


def run_gradient_viz(loss_type="jtfs", time_shift=None):
    f0 = torch.tensor([256], dtype=torch.float32, requires_grad=False).cuda().reshape(1,1)
    fm1 = torch.tensor([4096], dtype=torch.float32, requires_grad=False).cuda().reshape(1,1)
    N = 20

    target_idx = N * (N // 2) + (N // 2)

    AM, FM = grid2d(x1=5, x2=25, y1=0.1, y2=1, n=N)
    X = AM.numpy().reshape((N, N))
    Y = FM.numpy().reshape((N, N))
    AM.requires_grad = True
    FM.requires_grad = True
    thetas = torch.stack([AM, FM], dim=-1).cuda()

    sr = 2**13
    duration = 2
    npartials = 128
    n_input = sr * duration

    theta_target = thetas[target_idx].clone().detach().requires_grad_(False)
    target = (
        ripple(
            [theta_target[1].reshape(1,1).cuda(), theta_target[0].reshape(1,1).cuda(), f0, fm1],
            sr=sr,
            duration=duration,
            n_partials=npartials, 
        )
        .cuda()
        .detach()
    )

    if loss_type == "jtfs":
        """
        loss_fn = TimeFrequencyScatteringLoss(
            shape=(n_input,),
            T=2**13,
            Q=(8, 2),
            J=12,
            J_fr=5,
            F=0,
            Q_fr=2,
            format="time",
        )
        Sx_target = loss_fn.ops[0](target.cuda()).detach()
        """
        loss_fn = WeightedTimeFrequencyScatteringLoss(
            shape=(n_input,), Q=(8, 2), J=12, J_fr=5, Q_fr=2, format="time", weights=[0.25, 1.0]
        )
        Sx_target = loss_fn.ops[0](target.cuda()).detach()[0]

    elif loss_type == "mss":
        loss_fn = MultiScaleSpectralLoss(max_n_fft=1024)

    x, y, u, v = [], [], [], []
    losses, grads = [], []
    for theta in tqdm(thetas):
        am = torch.tensor(theta[0], requires_grad=True, dtype=torch.float32).reshape(1,1).cuda()
        fm = torch.tensor(theta[1], requires_grad=True, dtype=torch.float32).reshape(1,1).cuda()
        am.retain_grad()
        fm.retain_grad()
        audio = ripple([fm, am, f0, fm1], sr=sr, duration=duration, n_partials=npartials)

        loss = (
            loss_fn(audio.cuda(), Sx_target.cuda(), transform_y=False)
            if loss_type == "jtfs"
            else loss_fn(audio, target)
        )
        loss.backward()
        losses.append(float(loss.detach().cpu().numpy()))
        x.append(float(am))
        y.append(float(fm))
        u.append(float(-am.grad))
        v.append(float(-fm.grad))

        grad = np.stack([float(-am.grad), float(-fm.grad)])
        grads.append(grad)

    zs = np.array(losses)
    Z = zs.reshape(X.shape)

    plot_contour_gradient(
        X,
        Y,
        Z,
        target_idx,
        grads,
        save_path=f"img/grad_field_{loss_type}_{time_shift}.png",
    )
    mesh_plot_3d(
        X, Y, Z, target_idx, save_path=f"img/3d_mesh_{loss_type}_{time_shift}.png"
    )


def main():
    fire.Fire(run_gradient_viz)


if __name__ == "__main__":
    main()
