import numpy as np, torch, random, fire
from tqdm import tqdm

from meso_dtfa.core import ripple, grid2d
from meso_dtfa.loss import TimeFrequencyScatteringLoss, MultiScaleSpectralLoss, TimeFrequencyScatteringS2Loss
from meso_dtfa.plot import plot_contour_gradient, mesh_plot_3d


def run_gradient_viz(loss_type="jtfs", time_shift=None):
    N = 20

    target_idx = N * (N // 2) + (N // 2)
    
    AM, FM = grid2d(x1=4, x2=16, y1=2, y2=4, n=N)
    X = AM.numpy().reshape((N, N))
    Y = FM.numpy().reshape((N, N))
    AM.requires_grad = True
    FM.requires_grad = True
    thetas = torch.stack([AM, FM], dim=-1)

    sr = 2**13
    duration = 4
    n_input = sr * duration
    n_partials = 128

    f0 = torch.tensor([250], dtype=torch.float32, requires_grad=False)[None, :]
    fm1 = torch.tensor([sr // 2], dtype=torch.float32, requires_grad=False)[None, :]
    theta_target = thetas[target_idx].detach().requires_grad_(False)

    target = ripple(theta=[torch.tensor([theta_target[0]])[None, :], 
                           torch.tensor([theta_target[1]])[None, :],
                           f0, fm1], 
                    duration=duration,
                    n_partials=n_partials,
                    sr=sr,
                    window=True).cuda().detach()

    if loss_type == "jtfs":
        loss_fn = TimeFrequencyScatteringLoss(
            shape=(n_input,), Q=(8, 2), J=12, J_fr=5, Q_fr=2, format="time", T=2**14
        )
        # idxs = np.where(loss_fn.ops[0].meta()['order'] == 2)
        Sx_target = loss_fn.ops[0](target.cuda()).detach()
    elif loss_type == "mss":
        loss_fn = MultiScaleSpectralLoss(max_n_fft=1024)

    x, y, u, v = [], [], [], []
    losses, grads = [], []
    for theta in tqdm(thetas):
        am = torch.tensor([[theta[0]]], requires_grad=True, dtype=torch.float32)
        fm = torch.tensor([[theta[1]]], requires_grad=True, dtype=torch.float32)
        audio = ripple(theta=[am, fm, f0, fm1], 
                       duration=duration,
                       n_partials=n_partials,
                       sr=sr, 
                       window=True)

        loss = loss_fn(audio.cuda(), Sx_target.cuda(), transform_y=False) if loss_type == "jtfs" else loss_fn(audio, target)
        loss.backward()
        losses.append(float(loss.detach().cpu().numpy()))
        x.append(float(am))
        y.append(float(fm))
        u.append(float(- am.grad))
        v.append(float(- fm.grad))

        grad = np.stack([float(- am.grad), float(-fm.grad)])
        grads.append(grad)

    zs = np.array(losses)
    Z = zs.reshape(X.shape)

    plot_contour_gradient(X, Y, Z, 
                          target_idx, 
                          grads, 
                          save_path=f"img/grad_field_ripple_{loss_type}_{time_shift}.png", 
                          ylabel="FM (octaves / second)")
    mesh_plot_3d(-X, Y, Z, 
                 target_idx, 
                 save_path=f"img/3d_mesh_ripple_{loss_type}_{time_shift}.png",
                 ylabel="FM (octaves / second)")



def main():
  fire.Fire(run_gradient_viz)

if __name__ == "__main__":
    main()