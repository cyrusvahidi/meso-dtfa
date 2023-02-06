import fire, numpy as np, torch, matplotlib.pyplot as plt

from kymatio.torch import TimeFrequencyScattering

from meso_dtfa.core import generate_am_chirp, grid2d
from meso_dtfa.loss import (
    MultiScaleSpectralLoss,
    TimeFrequencyScatteringLoss,
    DistanceLoss,
)


def run_gradient_descent(loss_type="jtfs", time_shift=10):

    f0 = torch.tensor([512], dtype=torch.float32, requires_grad=False).cuda()
    N = 20

    target_idx = N * (N // 2) + (N // 2)

    AM, FM = grid2d(x1=4, x2=16, y1=0.5, y2=4, n=N)
    thetas = torch.stack([AM, FM], dim=-1).cuda()  # (400, 2)

    sr = 2**13
    duration = 4
    n_input = sr * duration

    jtfs_kwargs = {
        "shape": (n_input,),
        "Q": (8, 2),
        "J": 12,
        "J_fr": 5,
        "F": 0,
        "Q_fr": 2,
        "format": "time",
    }
    jtfs = TimeFrequencyScattering(**jtfs_kwargs).cuda()

    if loss_type == "jtfs":
        specloss = TimeFrequencyScatteringLoss(**jtfs_kwargs)
        lr = 100
    elif loss_type == "mss":
        specloss = MultiScaleSpectralLoss()
        lr = 1e-4

    Ploss = DistanceLoss()

    theta_target = thetas[target_idx].clone().detach().requires_grad_(False)
    # define time shift
    target = (
        generate_am_chirp(
            [f0, theta_target[0], theta_target[1]],
            sr=sr,
            duration=duration,
            delta=2**time_shift,
        )
        .cuda()
        .detach()
    )
    Sx_target = jtfs(target.cuda()).detach()

    slosses, plosses, grads = [], [], []
    best_ploss = np.inf
    # start at a random initial prediction
    iters = 100
    pred_idx = np.random.choice(np.arange(N * N))
    theta_prediction = thetas[target_idx + 10].clone().detach().requires_grad_(False)
    am = torch.tensor(theta_prediction[0], requires_grad=True, dtype=torch.float32)
    fm = torch.tensor(theta_prediction[1], requires_grad=True, dtype=torch.float32)
    print("initial prediction", am, fm)

    for iter in range(iters):
        audio = generate_am_chirp([f0, am, fm], sr=sr, duration=duration)
        if loss_type == "jtfs":
            sloss = specloss(audio, Sx_target, transform_y=False)
        elif loss_type == "mss":
            sloss = specloss(audio, target, transform_y=True)
        sloss.backward()
        slosses.append(float(sloss.detach().cpu().numpy()))
        ploss = Ploss.dist(torch.tensor([am, fm]).cuda(), theta_target.cuda())
        plosses.append(float(ploss.cpu().numpy()))
        grad = np.stack([float(-am.grad), float(-fm.grad)])
        grads.append(grad)
        print(
            f" update | iter {iter}| lr {lr} | AM pred={float(am):.3f}, target={float(theta_target[0]):.3f} | FM pred={float(fm):.3f}, target={float(theta_target[1]):.3f}, ploss={float(ploss):.3f}"
        )

        if plosses[-1] <= best_ploss:
            best_ploss = plosses[-1]
            am_opt = am.clone()
            fm_opt = fm.clone()
            lr *= 1.2
            with torch.no_grad():
                am = am - am.grad * lr
                fm = fm - fm.grad * lr
                am.requires_grad = True
                fm.requires_grad = True
        else:
            with torch.no_grad():
                am = torch.tensor(am_opt.clone(), requires_grad=True)
                fm = torch.tensor(fm_opt.clone(), requires_grad=True)
            lr *= 0.5

    fig = plt.figure()
    plt.plot(np.arange(iters), plosses)
    plt.title(
        "Gradient descent with {} loss time shifted by 2**{} samples".format(
            loss_type, time_shift
        )
    )
    plt.xlabel("iterations")
    plt.ylabel("L2 parameter loss")
    plt.show()
    plt.savefig("{} losses with 2**{} time shift.png".format(loss_type, time_shift))

    np.save("{}_ts{}_ploss.npy".format(loss_type, time_shift), plosses)


def main():
    fire.Fire(run_gradient_descent)


if __name__ == "__main__":
    main()
