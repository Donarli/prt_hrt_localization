import numpy as np
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from signal_model.gemini_grad import f
from config import C

def run_smoke_test():
    np.random.seed(42)
    
    # 1. Setup problem dimensions and synthetic data
    N, M = 25, 4
    A = np.random.randn(N, M) + 1j * np.random.randn(N, M)
    r = 12.5
    theta = 45.0
    f_c = 1e9
    xy_pos = np.stack([np.linspace(-12, 12, 25), np.zeros(25)], axis = 1) * 3e8/f_c/2
    c = C
    k = 2.0 * np.pi * f_c / c

    # Objective function scalar J(r, theta)
    def objective_J(r_val, th_val):
        th_rad = np.radians(th_val)
        dx = r_val * np.sin(th_rad) - xy_pos[:, 0]
        dy = r_val * np.cos(th_rad) - xy_pos[:, 1]
        d = np.sqrt(dx**2 + dy**2)
        a = np.exp(-1j * k * d)
        w = A.T.conj() @ a
        return np.real(np.vdot(w, w))

    # 2. Compute analytical gradient and Hessian
    H_ana, g_ana = f(A, r, theta, xy_pos, f_c)

    # 3. Finite Difference Gradient Check (using optimal step size eps=1e-4)
    eps_r = 1e-6
    eps_th = 1e-4
    g_num_r = (objective_J(r + eps_r, theta) - objective_J(r - eps_r, theta)) / (2 * eps_r)
    g_num_th = (objective_J(r, theta + eps_th) - objective_J(r, theta - eps_th)) / (2 * eps_th)
    g_num = np.array([[g_num_r], [g_num_th]])

    # 4. Finite Difference Hessian Check
    eps_h = 1e-5
    _, g_r_plus = f(A, r + eps_h, theta, xy_pos, f_c)
    _, g_r_minus = f(A, r - eps_h, theta, xy_pos, f_c)
    dg_dr = (g_r_plus - g_r_minus) / (2 * eps_h)

    _, g_th_plus = f(A, r, theta + eps_h, xy_pos, f_c)
    _, g_th_minus = f(A, r, theta - eps_h, xy_pos, f_c)
    dg_dth = (g_th_plus - g_th_minus) / (2 * eps_h)

    H_num = np.hstack([dg_dr, dg_dth])

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