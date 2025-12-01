import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov as lyap
from scipy.linalg import eigvals
from scipy.signal import place_poles

plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0


def main():
    # System Definition
    np_dim, nc = 2, 2
    n = np_dim + nc

    A_p = np.array([[-6.0, 2.0], [-5.0, -1.0]])
    A_c = np.array([[-7.0, 4.0], [-8.0, -7.0]])
    B_p = np.array([[1.0], [1.0]])
    B_c = np.array([[1.0], [1.0]])
    C_c = np.array([[1.0, 0.0]])
    D_c = 1.0

    A = np.block([[A_p, B_p @ C_c], [np.zeros((nc, np_dim)), A_c]])
    B = np.vstack([B_p * D_c, B_c])

    Q_p = 0.5 * np.eye(np_dim)
    Q = np.block(
        [[Q_p, np.zeros((np_dim, nc))], [np.zeros((nc, np_dim)), np.zeros((nc, nc))]]
    )

    pi_v = np.array([[1.0], [-3.0]])

    # The gamma bound
    Y = 0.2 * np.eye(n)
    S = lyap(A.T, -Y)

    norm_SB = la.norm(S @ B, 2)
    norm_Qppi = la.norm(Q_p @ pi_v, 2)
    lam_min_Y = np.min(la.eigvals(Y))
    gamma_max = lam_min_Y / (4.0 * (norm_SB) * (norm_Qppi))

    gamma = 0.9 * gamma_max

    # Scaling
    pi_v = gamma * pi_v

    Hbar = np.hstack([2.0 * (pi_v.T @ Q_p), np.zeros((1, nc))])
    Fbar = A + B @ Hbar

    eigA = la.eigvals(A)
    alphaA = float(np.max(np.real(eigA)))
    poles_hat = (alphaA - 6.0) + np.arange(0, n) * (-1.0)

    temp = place_poles(Fbar.T, Hbar.T, poles_hat)
    kd = temp.gain_matrix
    K = -kd.T
    L = K - B

    eigZhat = la.eigvals(Fbar + (B + L) @ Hbar)

    def h(z: np.ndarray) -> float:
        """The output map"""
        return float((z.T @ Q @ z).item())

    def z_zhat_dynamics(z: np.ndarray, zhat: np.ndarray):
        """Plant and observer dynamics."""
        hz = h(z)
        hzhat = h(zhat)
        Hzhat = float((Hbar @ zhat).item())

        zdot = A @ z + B * (hz + Hzhat)
        zhatdot = A @ zhat + B * (hzhat + 2.0 * Hzhat) + L * (hzhat - hz + Hzhat)
        return zdot.reshape(-1, 1), zhatdot.reshape(-1, 1)

    def rk4_step(z: np.ndarray, zhat: np.ndarray, dt: float):
        """4th-order Runge-Kutta integration step."""
        k1z, k1zh = z_zhat_dynamics(z, zhat)
        k2z, k2zh = z_zhat_dynamics(z + 0.5 * dt * k1z, zhat + 0.5 * dt * k1zh)
        k3z, k3zh = z_zhat_dynamics(z + 0.5 * dt * k2z, zhat + 0.5 * dt * k2zh)
        k4z, k4zh = z_zhat_dynamics(z + dt * k3z, zhat + dt * k3zh)
        z_next = z + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
        zhat_next = zhat + (dt / 6.0) * (k1zh + 2 * k2zh + 2 * k3zh + k4zh)
        return z_next, zhat_next

    # Simulation
    T = 5.0
    dt = 1e-3
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)

    z0 = np.array([[0.1], [-0.15], [0.1], [-0.1]])
    zhat0 = np.array([[-0.1], [0.1], [-0.1], [0.1]])

    z, zhat = z0.copy(), zhat0.copy()
    Z, Zhat = np.zeros((N + 1, n)), np.zeros((N + 1, n))
    Z[0, :], Zhat[0, :] = z.ravel(), zhat.ravel()

    for k in range(1, N + 1):
        z, zhat = rk4_step(z, zhat, dt)
        Z[k, :], Zhat[k, :] = z.ravel(), zhat.ravel()

    # Plotting
    if not plt.get_backend().startswith("agg"):
        import os

        if not os.path.exists("figures"):
            os.makedirs("figures")

        true_style = dict(linewidth=1.5)
        estim_style = dict(linestyle="--", linewidth=1.5)
        state_labels = [r"$z_1$", r"$z_2$", r"$z_3$", r"$z_4$"]
        columnwidth_pt = 251.80688
        columnwidth_inches = columnwidth_pt / 72.0
        fig_height = columnwidth_inches

        # Plot 1: Trajectories
        fig, axes = plt.subplots(
            2, 2, figsize=(columnwidth_inches, fig_height), sharex=True
        )
        fig.subplots_adjust(wspace=0.6)
        axes_flat = axes.flatten()

        for i in range(n):
            ax = axes_flat[i]
            ax.plot(t, Z[:, i], **true_style, label="true")
            ax.plot(t, Zhat[:, i], **estim_style, label="estimate", zorder=10)
            ax.set_ylabel(state_labels[i])
            if i >= 2:
                ax.set_xlabel("Time [s]")
            ax.tick_params(axis="both")
            ax.grid(False)

        plt.savefig("figures/estimator.pdf", dpi=300, bbox_inches="tight")

        # Plot 2: Norms
        err = Zhat - Z
        norm_z = np.linalg.norm(Z, axis=1)
        norm_e = np.linalg.norm(err, axis=1)

        comp_fig_height = columnwidth_inches * 0.3
        fig, axes = plt.subplots(1, 1, figsize=(columnwidth_inches, comp_fig_height))

        axes.plot(t, norm_z, label="Norm(z)")
        axes.plot(t, norm_e, label="Norm(e)", linestyle="--")
        axes.set_xlabel("Time [s]")
        axes.set_ylabel("Magnitude")
        axes.grid(False)

        plt.savefig("figures/comparison.pdf", dpi=300, bbox_inches="tight")
        # plt.show()


if __name__ == "__main__":
    main()
