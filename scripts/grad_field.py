import numpy as np, torch, random, fire
from tqdm import tqdm

from meso_dtfa.core import generate_am_chirp, grid2d
from meso_dtfa.loss import TimeFrequencyScatteringLoss, MultiScaleSpectralLoss
from meso_dtfa.plot import plot_contour_gradient, mesh_plot_3d


def run_gradient_viz(loss_type="jtfs", time_shift=None):
    f0 = torch.tensor([512.0], dtype=torch.float32, requires_grad=False).cuda()
    N = 20

    target_idx = N * (N // 2) + (N // 2)

    AM, FM = grid2d(x1=4, x2=15, y1=0.5, y2=4, n=N)
    X = AM.numpy().reshape((N, N))
    Y = FM.numpy().reshape((N, N))
    AM.requires_grad = True
    FM.requires_grad = True
    thetas = torch.stack([AM, FM], dim=-1).cuda()

    sr = 2**13
    duration = 4
    n_input = sr * duration

    theta_target = thetas[target_idx].clone().detach().requires_grad_(False)
    target = (
        generate_am_chirp(
            [f0, theta_target[0], theta_target[1]], sr=sr, duration=duration
        )
        .cuda()
        .detach()
    )

    if loss_type == "jtfs":
        loss_fn = TimeFrequencyScatteringLoss(
            shape=(n_input,),
            #T=2**13,
            Q=(8, 2),
            J=12,
            J_fr=5,
            F="global",
            Q_fr=2,
            format="time",
        )
        Sx_target = loss_fn.ops[0](target.cuda()).detach()
    elif loss_type == "mss":
        loss_fn = MultiScaleSpectralLoss(max_n_fft=1024)

    x, y, u, v = [], [], [], []
    losses, grads = [], []
    for theta in tqdm(thetas):
        am = torch.tensor(theta[0], requires_grad=True, dtype=torch.float32)
        fm = torch.tensor(theta[1], requires_grad=True, dtype=torch.float32)
        audio = generate_am_chirp(
            [torch.tensor([768.0], dtype=torch.float32, requires_grad=False).cuda(), am, fm],
            sr=sr,
            duration=duration,
            delta=(2 ** random.randint(8, 12) if time_shift == "random" else 2**8)
            if time_shift
            else 0,
        )

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
