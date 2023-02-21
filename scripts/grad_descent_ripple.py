import fire, numpy as np, torch, matplotlib.pyplot as plt

from kymatio.torch import TimeFrequencyScattering

from meso_dtfa.core import ripple, grid2d
from meso_dtfa.loss import (
    MultiScaleSpectralLoss,
    TimeFrequencyScatteringLoss,
    DistanceLoss,
)


def run_gradient_descent(loss_type="jtfs"):

    f0 = torch.tensor([256], dtype=torch.float32, requires_grad=False).cuda().reshape(1,1)
    fm1 = torch.tensor([4096], dtype=torch.float32, requires_grad=False).cuda().reshape(1,1)
    N = 20

    target_idx = N * (N // 2) + (N // 2)

    AM, FM = grid2d(x1=2, x2=14, y1=2, y2=4, n=N)
    thetas = torch.stack([AM, FM], dim=-1).cuda()  # (400, 2)

    sr = 2**13
    duration = 2
    npartials = 128
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

    theta_target = thetas[target_idx].clone().detach().requires_grad_(False).cuda()
    # define time shift
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
    Sx_target = jtfs(target.cuda()).detach()

    slosses, plosses, grads = [], [], []
    best_ploss = np.inf
    # start at a random initial prediction
    iters = 100
    pred_idx = np.arange(0, N*N, N) + np.arange(0, N) #uniformly vary the prediction
    for i, pred in enumerate(pred_idx):
        slosses, plosses, grads = [], [], []
        best_ploss = np.inf
        theta_prediction = thetas[pred,:].clone().detach().requires_grad_(False)
        print("caculating {}".format(pred))
        am = torch.tensor([theta_prediction[0]], requires_grad=True, dtype=torch.float32).reshape(1,1).cuda()
        fm = torch.tensor([theta_prediction[1]], requires_grad=True, dtype=torch.float32).reshape(1,1).cuda()
        print("initial prediction", am, fm)

        for iter in range(iters):
            am.retain_grad()
            fm.retain_grad()
            audio = ripple([fm, am, f0, fm1], sr=sr, duration=duration, n_partials=npartials)
            if loss_type == "jtfs":
                sloss = specloss(audio, Sx_target, transform_y=False)
            elif loss_type == "mss":
                sloss = specloss(audio, target, transform_y=True)
            sloss.backward()
            slosses.append(float(sloss.detach().cpu().numpy()))
            ploss = Ploss.dist(torch.tensor([am, fm]).cuda(), theta_target.cuda())
            plosses.append(float(ploss.cpu().numpy()))
            #print("check grad", am.grad, fm.grad, fm1.grad, f0.grad)
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

        plt.rcParams.update({
                        "font.family": "serif",
                        'font.size': 11,
                        'text.usetex': False,
                        'axes.labelsize': 10
                        })
        fig = plt.figure()
        plt.plot(np.arange(iters), plosses)
        plt.xlabel("SGD iteration")
        plt.ylabel("AM/FM L2 distance")
        plt.show()
        plt.savefig("{} losses initialized at am={} fm={}.png".format(loss_type, thetas[pred][0], thetas[pred][1]))

        np.save("{}_ploss_init_{}.npy".format(loss_type, i), plosses)



def main():
    fire.Fire(run_gradient_descent)
    #run_gradient_descent("jtfs")

if __name__ == "__main__":
    main()
