import numpy as np
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from signal_model.gemini_grad import f_RTheTau
from config import C, SYSTEM


def gaussian_pulse(t, sigma=SYSTEM.gaussian_sigma):
    return np.exp(-0.5 * (t / sigma) ** 2).astype(np.complex128)


def run_smoke_test():
    np.random.seed(42)
    
    # # 1. Setup problem dimensions and synthetic data
    # N, M    = 513, 701
    # A       = np.random.randn(N, M) + 1j * np.random.randn(N, M)
    # r       = 55
    # theta   = 45.0
    # f_c     = SYSTEM.fc
    # delta_T = SYSTEM.dt
    # sigma   = SYSTEM.gaussian_sigma
    # xy_pos  = np.stack([np.linspace(-(N-1)/2, (N-1)/2, N), np.zeros(N)], axis = 1) * C/f_c/2
    # c       = C
    # k       = 2.0 * np.pi * f_c / c
    # dx      = r * np.sin( np.radians(theta)) - xy_pos[:, 0]
    # dy      = r * np.cos( np.radians(theta)) - xy_pos[:, 1]
    # d       = np.sqrt(dx**2 + dy**2)
    # tau     = - np.mean(d)/c

    # 1. Parameter configuration
    N, M = 513, 701
    A = np.random.randn(N, M) + 1j * np.random.randn(N, M)

    r = 100.0          # Distance in meters
    theta = 30.0      # Angle in degrees
    f_c = SYSTEM.fc         # Carrier frequency: 1 GHz
    delta_T = SYSTEM.dt    # Sampling period: 1 ns
    sigma = SYSTEM.gaussian_sigma     # Pulse width: 10 ns
    c  = C
    k  = 2.0 * np.pi * f_c / c
    xy_pos  = np.stack([np.linspace(-(N-1)/2, (N-1)/2, N), np.zeros(N)], axis = 1) * C/f_c/2
    dx = r * np.sin(np.radians(theta)) - xy_pos[:, 0]
    dy = r * np.cos(np.radians(theta)) - xy_pos[:, 1]
    d  = np.sqrt(dx**2 + dy**2)
    tau = -np.mean(d)/c        # Time delay in seconds (1 ns)


    # Objective function scalar J(r, theta)
    def objective_J(r_val, th_val, tau_val):
        # th_rad = np.radians(th_val)
        # dx = r_val * np.sin(th_rad) - xy_pos[:, 0]
        # dy = r_val * np.cos(th_rad) - xy_pos[:, 1]
        # d = np.sqrt(dx**2 + dy**2)
        # m_idx = np.arange(-(M-1)/2, (M-1)/2 + 1)
        # T_nm = m_idx[None, :] * delta_T - tau_val - (d[:, None] / c)

        # # Gaussian pulse s(T) and derivatives
        # s_nm = np.exp(-0.5 * (T_nm / sigma)**2)

        # if 0: 
        #     import matplotlib.pyplot as plt 
        #     plt.figure
        #     plt.plot(s_nm.T[:, 0])
        #     plt.grid()
        #     plt.savefig('figures/debug_p2.png')
        #     plt.close()
        #     breakpoint()

        # a = np.exp(-1j * k * d)        
        # return np.abs(np.sum(A.conj() * (a[:, None] * s_nm)))**2
        th_rad = np.radians(th_val)
        dx = r_val * np.sin(th_rad) - xy_pos[:, 0]
        dy = r_val * np.cos(th_rad) - xy_pos[:, 1]
        d = np.sqrt(dx**2 + dy**2)

        tau_n = d / c
        a = np.exp(-1j * k * d)

        m_idx = np.arange(-(M-1)/2, (M-1)/2 + 1)
        T_nm = m_idx[None, :] * delta_T - tau_val - tau_n[:, None]
        s_nm = np.exp(-0.5 * (T_nm / sigma)**2)

        B = a[:, None] * s_nm
        z = np.sum(np.conj(A) * B)
        return np.real(z * np.conj(z))


    # 2. Compute analytical gradient and Hessian
    H_ana, g_ana = f_RTheTau(A, r, theta, tau, xy_pos, f_c, delta_T, sigma)

    # 3. Finite Difference Gradient Check (using optimal step size eps=1e-4)
    eps_r = 1e-4*r
    eps_th = 1e-4*theta
    eps_tau = 1e-6*np.abs(tau)
    g_num_r = (objective_J(r + eps_r, theta, tau) - objective_J(r - eps_r, theta, tau)) / (2 * eps_r)
    g_num_th = (objective_J(r, theta + eps_th, tau) - objective_J(r, theta - eps_th, tau)) / (2 * eps_th)
    g_num_tau = (objective_J(r, theta, tau + eps_tau) - objective_J(r, theta, tau - eps_tau)) / (2 * eps_tau)
    g_num = np.array([[g_num_r], [g_num_th], [g_num_tau]])

    # g_num_r = (objective_J(r + eps_r, theta, tau) - objective_J(r - eps_r, theta, tau)) / (2 * eps_r)
    # g_num_th = (objective_J(r, theta + eps_th, tau) - objective_J(r, theta - eps_th, tau)) / (2 * eps_th)
    # g_num_tau = (objective_J(r, theta, tau + eps_tau) - objective_J(r, theta, tau - eps_tau)) / (2 * eps_tau)
    # g_num = np.array([[g_num_r], [g_num_th], [g_num_tau]])



    # 4. Finite Difference Hessian Check
    _, g_r_plus = f_RTheTau(A, r + eps_r, theta, tau, xy_pos,  f_c, delta_T, sigma)
    _, g_r_minus = f_RTheTau(A, r - eps_r, theta, tau, xy_pos,  f_c, delta_T, sigma)
    dg_dr = (g_r_plus - g_r_minus) / (2 * eps_r)

    _, g_th_plus = f_RTheTau(A, r, theta + eps_th, tau, xy_pos,  f_c, delta_T, sigma)
    _, g_th_minus = f_RTheTau(A, r, theta - eps_th, tau, xy_pos,  f_c, delta_T, sigma)
    dg_dth = (g_th_plus - g_th_minus) / (2 * eps_th)

    _, g_tau_plus = f_RTheTau(A, r, theta, tau + eps_tau, xy_pos,  f_c, delta_T, sigma)
    _, g_tau_minus = f_RTheTau(A, r, theta, tau - eps_tau, xy_pos,  f_c, delta_T, sigma)
    dg_dtau = (g_tau_plus - g_tau_minus) / (2 * eps_tau)

    H_num = np.hstack([dg_dr, dg_dth, dg_dtau])

    # Relative Errors
    g_rel_err = np.linalg.norm(g_ana - g_num) / np.linalg.norm(g_ana)
    H_rel_err = np.linalg.norm(H_ana - H_num) / np.linalg.norm(H_ana)

    print("--- Gradient Comparison ---")
    print("Analytical:\n", g_ana)
    print("Numerical:\n", g_num)
    print(f"Gradient Relative Error: {g_rel_err:.2e}\n")

    print("--- Hessian Comparison ---")
    print("Analytical:\n", H_ana)
    print("Numerical:\n", H_num)
    print(f"Hessian Relative Error:  {H_rel_err:.2e}\n")

    assert g_rel_err < 1e-4, f"Gradient test failed with error {g_rel_err}"
    assert H_rel_err < 1e-4, f"Hessian test failed with error {H_rel_err}"
    print("[SUCCESS] Smoke test passed!")

if __name__ == "__main__":
    run_smoke_test()