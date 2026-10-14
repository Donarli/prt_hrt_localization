import numpy as np
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from signal_model.gemini_grad import compute_gradient_hessian_3d_ratio
from config import C, SYSTEM


def compute_gradient_hessian_3d_ratio(A, r, theta, tau_ratio, xy_pos, f_c, t, delta_T, sigma):
    """
    Computes 3x1 gradient and 3x3 Hessian of J(r, theta, tau_ratio) = |trace(A.H @ B)|^2,
    where tau_ratio = tau / delta_T is a relative dimensionless delay.
    """
    c = 3e8
    k = 2.0 * np.pi * f_c / c
    deg2rad = np.pi / 180.0

    # Geometry & steering vector
    th_rad = np.radians(theta)
    sin_th, cos_th = np.sin(th_rad), np.cos(th_rad)

    x_i, y_i = xy_pos[:, 0], xy_pos[:, 1]
    dx = r * sin_th - x_i
    dy = r * cos_th - y_i
    d = np.sqrt(dx**2 + dy**2)

    a = np.exp(-1j * k * d)

    # Distance derivatives
    dd_dr = (dx * sin_th + dy * cos_th) / d
    dd_dth = deg2rad * r * (dx * cos_th - dy * sin_th) / d

    d2d_dr2 = (1.0 - dd_dr**2) / d
    d2d_drdth = - (dd_dr * dd_dth) / d + dd_dth / r
    d2d_dth2 = - (dd_dth**2) / d + (deg2rad**2 * r**2) / d - deg2rad**2 * r * dd_dr

    # Time indices & argument matrix T_nm (N, M)
    N, M = A.shape
    m_idx = np.arange(M)
    T_nm = t - (m_idx[None, :] + tau_ratio) * delta_T - (d[:, None] / c)

    # Gaussian pulse s(T) and derivatives
    s_nm = np.exp(-0.5 * (T_nm / sigma)**2)
    ds_nm = - (T_nm / (sigma**2)) * s_nm
    d2s_nm = ((T_nm**2 - sigma**2) / (sigma**4)) * s_nm

    # Complex matrix B
    B = a[:, None] * s_nm
    z = np.sum(np.conj(A) * B)

    # Intermediate combination factors
    P_nm = -1j * k * s_nm - (1.0 / c) * ds_nm
    Q_nm = - (k**2) * s_nm + (2j * k / c) * ds_nm + (1.0 / (c**2)) * d2s_nm
    R_nm = 1j * k * ds_nm + (1.0 / c) * d2s_nm

    # 1st derivatives of B_nm wrt [r, theta, tau_ratio]
    dB_dr = dd_dr[:, None] * a[:, None] * P_nm
    dB_dth = dd_dth[:, None] * a[:, None] * P_nm
    dB_dtau_ratio = - a[:, None] * ds_nm * delta_T  # Chain rule factor: * delta_T

    # 1st derivatives of z
    dz_dr = np.sum(np.conj(A) * dB_dr)
    dz_dth = np.sum(np.conj(A) * dB_dth)
    dz_dtau_ratio = np.sum(np.conj(A) * dB_dtau_ratio)

    grad_z = np.array([[dz_dr], [dz_dth], [dz_dtau_ratio]], dtype=np.complex128)

    # 2nd derivatives of B_nm
    d2B_dr2 = d2d_dr2[:, None] * a[:, None] * P_nm + (dd_dr[:, None]**2) * a[:, None] * Q_nm
    d2B_drdth = d2d_drdth[:, None] * a[:, None] * P_nm + (dd_dr[:, None] * dd_dth[:, None]) * a[:, None] * Q_nm
    d2B_dth2 = d2d_dth2[:, None] * a[:, None] * P_nm + (dd_dth[:, None]**2) * a[:, None] * Q_nm

    d2B_drdtau_ratio = dd_dr[:, None] * a[:, None] * R_nm * delta_T
    d2B_dthdtau_ratio = dd_dth[:, None] * a[:, None] * R_nm * delta_T
    d2B_dtau_ratio2 = a[:, None] * d2s_nm * (delta_T**2)

    # 2nd derivatives of z
    d2z_dr2 = np.sum(np.conj(A) * d2B_dr2)
    d2z_drdth = np.sum(np.conj(A) * d2B_drdth)
    d2z_dth2 = np.sum(np.conj(A) * d2B_dth2)
    d2z_drdtau_ratio = np.sum(np.conj(A) * d2B_drdtau_ratio)
    d2z_dthdtau_ratio = np.sum(np.conj(A) * d2B_dthdtau_ratio)
    d2z_dtau_ratio2 = np.sum(np.conj(A) * d2B_dtau_ratio2)

    hess_z = np.array([
        [d2z_dr2,          d2z_drdth,        d2z_drdtau_ratio],
        [d2z_drdth,        d2z_dth2,         d2z_dthdtau_ratio],
        [d2z_drdtau_ratio, d2z_dthdtau_ratio, d2z_dtau_ratio2]
    ], dtype=np.complex128)

    # Real Gradient (3x1) and Hessian (3x3)
    gradient = 2.0 * np.real(np.conj(z) * grad_z)
    hessian = 2.0 * np.real(np.conj(z) * hess_z + grad_z @ grad_z.T.conj())

    return hessian, gradient

def run_smoke_test_ratio():
    np.random.seed(42)
    N, M = 4, 3
    A = np.random.randn(N, M) + 1j * np.random.randn(N, M)
    xy_pos = np.random.randn(N, 2)

    r = 10.0          
    theta = 30.0      
    tau_ratio = 1.0   # Dimensionless ratio (tau = 1.0 * delta_T = 1 ns)
    f_c = 1e9         
    t = 35e-9         
    delta_T = 1e-9    
    sigma = 10e-9     
    c = 3e8
    k = 2.0 * np.pi * f_c / c

    def objective_J(r_val, th_val, tr_val):
        th_rad = np.radians(th_val)
        dx = r_val * np.sin(th_rad) - xy_pos[:, 0]
        dy = r_val * np.cos(th_rad) - xy_pos[:, 1]
        d = np.sqrt(dx**2 + dy**2)

        tau_n = d / c
        a = np.exp(-1j * k * d)

        m_idx = np.arange(M)
        T_nm = t - (m_idx[None, :] + tr_val) * delta_T - tau_n[:, None]
        s_nm = np.exp(-0.5 * (T_nm / sigma)**2)

        B = a[:, None] * s_nm
        z = np.sum(np.conj(A) * B)
        return np.real(z * np.conj(z))

    # Analytical
    H_ana, g_ana = compute_gradient_hessian_3d_ratio(A, r, theta, tau_ratio, xy_pos, f_c, t, delta_T, sigma)

    # Finite Differences (a uniform step size eps=1e-5 works across all parameters!)
    eps = 1e-5
    g_num_r = (objective_J(r + eps, theta, tau_ratio) - objective_J(r - eps, theta, tau_ratio)) / (2 * eps)
    g_num_th = (objective_J(r, theta + eps, tau_ratio) - objective_J(r, theta - eps, tau_ratio)) / (2 * eps)
    g_num_tr = (objective_J(r, theta, tau_ratio + eps) - objective_J(r, theta, tau_ratio - eps)) / (2 * eps)
    g_num = np.array([[g_num_r], [g_num_th], [g_num_tr]])

    # Hessian Finite Difference
    _, g_r_p = compute_gradient_hessian_3d_ratio(A, r + eps, theta, tau_ratio, xy_pos, f_c, t, delta_T, sigma)
    _, g_r_m = compute_gradient_hessian_3d_ratio(A, r - eps, theta, tau_ratio, xy_pos, f_c, t, delta_T, sigma)
    dg_dr = (g_r_p - g_r_m) / (2 * eps)

    _, g_th_p = compute_gradient_hessian_3d_ratio(A, r, theta + eps, tau_ratio, xy_pos, f_c, t, delta_T, sigma)
    _, g_th_m = compute_gradient_hessian_3d_ratio(A, r, theta - eps, tau_ratio, xy_pos, f_c, t, delta_T, sigma)
    dg_dth = (g_th_p - g_th_m) / (2 * eps)

    _, g_tr_p = compute_gradient_hessian_3d_ratio(A, r, theta, tau_ratio + eps, xy_pos, f_c, t, delta_T, sigma)
    _, g_tr_m = compute_gradient_hessian_3d_ratio(A, r, theta, tau_ratio - eps, xy_pos, f_c, t, delta_T, sigma)
    dg_dtr = (g_tr_p - g_tr_m) / (2 * eps)

    H_num = np.hstack([dg_dr, dg_dth, dg_dtr])

    g_rel_err = np.linalg.norm(g_ana - g_num) / np.linalg.norm(g_ana)
    H_rel_err = np.linalg.norm(H_ana - H_num) / np.linalg.norm(H_ana)

    print("--- Dimensionless 3D Gradient ---")
    print("Analytical:\n", g_ana)
    print("Numerical:\n", g_num)
    print(f"Gradient Relative Error: {g_rel_err:.2e}\n")

    print("--- Dimensionless 3D Hessian ---")
    print("Analytical:\n", H_ana)
    print("Numerical:\n", H_num)
    print(f"Hessian Relative Error:  {H_rel_err:.2e}\n")

    assert g_rel_err < 1e-4, "Gradient check failed!"
    assert H_rel_err < 1e-4, "Hessian check failed!"
    print("[SUCCESS] Dimensionless 3D smoke test passed!")

if __name__ == "__main__":
    run_smoke_test_ratio()