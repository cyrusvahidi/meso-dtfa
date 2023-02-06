import os, matplotlib.pyplot as plt, numpy as np, librosa, librosa.display
import matplotlib.transforms as mtransforms
from matplotlib import cm


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "text.usetex": False,
        "axes.labelsize": 11,
    }
)


def plot_gradient_field(x, y, u, v, x_range, y_range, target, save_path):
    plt.figure()

    plt.scatter(x, y, color="r")

    u = np.array(u) / np.max(np.abs(u))  # gradient wrt AM
    v = np.array(v) / np.max(np.abs(v))  # gradient wrt FM
    grads = np.stack([u, v])
    grads = grads / (np.linalg.norm(grads, axis=0) + 1e-8)
    plt.quiver(x, y, grads[0, :], grads[1, :])

    plt.scatter([target[0]], [target[1]], color="g")

    plt.xticks(np.arange(x_range[0], x_range[1] + 1))
    plt.yticks(np.arange(y_range[0], y_range[1] + 1))
    plt.xlabel("AM (Hz)")
    plt.ylabel("FM (oct / s)")
    plt.show()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)


def plot_contour_gradient(X, Y, Z, target_idx, grads, save_path):
    """
    X, Y, Z: meshgrid (N, N) matrices
    target_idx: index of the target to scatter in green
    grads: list of gradients [u, v]
    save_path: where to save
    """
    fig, ax = plt.subplots()
    ax.contourf(X, Y, Z, 20, cmap=cm.coolwarm)

    target = [X.ravel()[target_idx], Y.ravel()[target_idx]]
    plt.scatter(target[0], target[1], color="g", alpha=1)

    grads = np.stack(grads)
    u = np.array(grads[:, 0]) / np.max(np.abs(grads[:, 0]))  # gradient wrt AM
    v = np.array(grads[:, 1]) / np.max(np.abs(grads[:, 1]))  # gradient wrt FM
    grads = np.stack([u, v])
    grads = grads / (np.linalg.norm(grads, axis=0) + 1e-8)
    ax.quiver(X.reshape(-1), Y.reshape(-1), grads[0, :], grads[1, :])

    # Plot Labelling
    plt.xlabel("AM (Hz)")
    ax.loglog()
    plt.ylabel("FM (oct / s)")
    plt.rcParams["axes.formatter.min_exponent"] = 2
    plt.show()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)


def mesh_plot_3d(X, Y, Z, target_idx, save_path):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    target = [X.ravel()[target_idx], Y.ravel()[target_idx]]

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.5)

    ax.scatter(target[0], target[1], color="g", alpha=1)

    ax.set_xlabel("AM (Hz)")
    ax.set_ylabel("FM (oct / s)")

    plt.show()
    plt.savefig(save_path)


def plot_spec(y, hop_length=256, n_fft=4096, sr=2**13):
    fig, ax = plt.subplots()
    D = librosa.stft(y, hop_length=hop_length, n_fft=n_fft)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    librosa.display.specshow(
        S_db, sr=sr, x_axis="time", y_axis="log", ax=ax, cmap="magma_r"
    )


def plot_scalogram(Sx, S, sr=22050):
    x_coords = librosa.times_like(Sx, hop_length=S.T)
    y_coords = [psi["xi"] * sr for psi in S.psi1_f]

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(
        Sx[1:, :].numpy(),
        sr=sr,
        x_coords=x_coords,
        x_axis="time",
        y_coords=y_coords,
        y_axis="cqt_hz",
        cmap="magma",
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, sr // 2)
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.minorticks_off()


def plot_jtfs(Sx, S, sr=22050):
    x_coords = librosa.times_like(Sx, hop_length=1)

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(
        Sx[1:, :], sr=sr, x_coords=x_coords, x_axis="time", cmap="magma"
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.xticks(np.arange(0, 1.5, 0.2))
    plt.minorticks_off()


def plot_jtfs_paths(jtfs, Sx):
    order2 = np.where(jtfs.meta()["order"] == 2)
    xi_frs = sorted(list(set(jtfs.meta()["xi_fr"][order2])), reverse=True)
    xis = sorted(list(set(jtfs.meta()["xi"][order2][:, 1])))
    s = Sx[np.where(jtfs.meta()["order"] == 2)]
    time = s.shape[-1]
    freq = s.shape[-2]
    paths = np.zeros((len(xi_frs), len(xis), freq, time))

    for i, s2 in enumerate(Sx[np.where(jtfs.meta()["order"] == 2)]):
        y = xi_frs.index(jtfs.meta()["xi_fr"][order2][i])
        x = xis.index(jtfs.meta()["xi"][order2][:, 1][i])

        paths[y, x, :, :] = s2.numpy()

    fig, axs = plt.subplots(
        len(xi_frs), len(xis), sharex=True, sharey=True, figsize=(30, 30)
    )
    axs[0, 0].invert_yaxis()
    for y in range(len(xi_frs)):
        for x in range(len(xis)):
            ax = axs[y, x]
            cmesh = librosa.display.specshow(
                paths[y, x], sr=sr, x_axis="time", cmap="magma", ax=ax
            )
            cmesh.set_clim(paths.min(), paths.max())
            ax.set_axis_off()

    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    for i, ax in enumerate(axs[:, 0]):
        trans = mtransforms.ScaledTranslation(-10 / 72, 0.75, fig.dpi_scale_trans)
        ax.text(
            0,
            0,
            f"{xi_frs[i]:.2f}",
            transform=ax.transAxes + trans,
            rotation=90,
            fontsize="large",
            ha="center",
            va="center",
            fontfamily="serif",
            weight="bold",
        )
    for i, ax in enumerate(axs[-1, :]):
        trans = mtransforms.ScaledTranslation(0.75, -0.25, fig.dpi_scale_trans)
        ax.text(
            0.0,
            0,
            f"{xis[i]:.4f}",
            transform=ax.transAxes + trans,
            fontsize="large",
            ha="center",
            va="center",
            fontfamily="serif",
            weight="bold",
        )

    fig.text(
        0.5,
        0.1,
        "temporal modulation (Hz)",
        fontsize="large",
        ha="center",
        va="center",
        weight="bold",
    )
    fig.text(
        0.1,
        0.5,
        "frequential modulation (cycles/octave)",
        fontsize="large",
        weight="bold",
        ha="center",
        va="center",
        rotation="vertical",
    )
    return paths
