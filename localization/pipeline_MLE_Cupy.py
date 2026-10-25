"""
MLE Method: 

The objective is to create the maximum correlation with received array samples 
Model-based received samples is controlled by 3 parameters: r, theta, t (time-of-arrival)

1. Produce Initial Guess (Either With Hint or With Structured Scanning)
2. Perform Newton's Method to obtain MLE estimate of the set of parameters

The initial Guess is produced either through radon transform or sequentially without radon
With "clean_rx", the post-refinement RMSE is on the level of 1e-10.

"""

from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np

os.environ.setdefault("CUPY_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".cupy_cache"))
try:
    import cupy as cp
except:
    import numpy as cp
    

from config import C, SYSTEM, ARRAY
from signal_model.propagation import propagation_delays
from localization.pipeline import LocalizationResult, localize

# ============================================================
# PRT settings
# ============================================================

PM_THETA_MIN, PM_THETA_MAX = -60.0, 60.0
PM_R_MIN, PM_R_MAX = 50.0, 400.0
PM_R_GRID          = cp.linspace(PM_R_MIN, PM_R_MAX, 60)

# ============================================================
# Space-time Beam settings
# ============================================================
PRNT  = False

# ============================================================
# Helpers
# ============================================================

def find_parabola_vertex(X, Y):
    """
    Calculates the center (vertex) of a vertical parabola passing through three points.
    
    Parameters:
    X (array-like): A 3x1 vector of x-coordinates [x1, x2, x3]
    Y (array-like): A 3x1 vector of y-coordinates [y1, y2, y3]
    
    Returns:
    tuple: (h, k) coordinates of the parabola's vertex.
    """
    # Unpack the 3x1 vector elements
    x1, x2, x3 = X[0], X[1], X[2]
    y1, y2, y3 = Y[0], Y[1], Y[2]
    
    # 1. Compute the horizontal coordinate (h)
    num_h = x1**2 * (y2 - y3) + x2**2 * (y3 - y1) + x3**2 * (y1 - y2)
    den_h = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    # Check for collinearity to prevent division by zero
    if den_h == 0:
        raise ValueError("The points are collinear or share x-coordinates; no unique parabola exists.")
        
    h = num_h / den_h
    
    # 2. Compute the vertical coordinate (k) using Lagrange Interpolation at x = h
    term1 = y1 * ((h - x2) * (h - x3)) / ((x1 - x2) * (x1 - x3))
    term2 = y2 * ((h - x1) * (h - x3)) / ((x2 - x1) * (x2 - x3))
    term3 = y3 * ((h - x1) * (h - x2)) / ((x3 - x1) * (x3 - x2))
    
    k = term1 + term2 + term3
    
    return h, k


def f_RTheTau(A, r, theta, tau_ratio, xy_pos, f_c, delta_T, sigma):
    """
    Computes 3x1 gradient and 3x3 Hessian of J(r, theta, tau_ratio) = |trace(A.H @ B)|^2,
    where tau_ratio = tau / delta_T is a relative dimensionless delay.
    """
    c = C
    k = 2.0 * cp.pi * f_c / c
    deg2rad = cp.pi / 180.0

    # Geometry & steering vector
    th_rad = cp.radians(theta)
    sin_th, cos_th = cp.sin(th_rad), cp.cos(th_rad)

    x_i, y_i = xy_pos[:, 0], xy_pos[:, 1]
    dx = r * sin_th - x_i
    dy = r * cos_th - y_i
    d = cp.sqrt(dx**2 + dy**2)

    a = cp.exp(-1j * k * d)

    # Distance derivatives
    dd_dr = (dx * sin_th + dy * cos_th) / d
    dd_dth = deg2rad * r * (dx * cos_th - dy * sin_th) / d

    d2d_dr2 = (1.0 - dd_dr**2) / d
    d2d_drdth = - (dd_dr * dd_dth) / d + dd_dth / r
    d2d_dth2 = - (dd_dth**2) / d + (deg2rad**2 * r**2) / d - deg2rad**2 * r * dd_dr

    # Time indices & argument matrix T_nm (N, M)
    N, M = A.shape
    m_idx = cp.arange(-(M-1)/2, (M-1)/2 + 1)
    T_nm = (m_idx[None, :] - tau_ratio) * delta_T - (d[:, None] / c)

    # Gaussian pulse s(T) and derivatives
    s_nm = cp.exp(-0.5 * (T_nm / sigma)**2)
    ds_nm = - (T_nm / (sigma**2)) * s_nm
    d2s_nm = ((T_nm**2 - sigma**2) / (sigma**4)) * s_nm

    # Complex matrix B
    B = a[:, None] * s_nm
    z = cp.sum(cp.conj(A) * B)

    # Intermediate combination factors
    P_nm = -1j * k * s_nm - (1.0 / c) * ds_nm
    Q_nm = - (k**2) * s_nm + (2j * k / c) * ds_nm + (1.0 / (c**2)) * d2s_nm
    R_nm = 1j * k * ds_nm + (1.0 / c) * d2s_nm

    # 1st derivatives of B_nm wrt [r, theta, tau_ratio]
    dB_dr = dd_dr[:, None] * a[:, None] * P_nm
    dB_dth = dd_dth[:, None] * a[:, None] * P_nm
    dB_dtau_ratio = - a[:, None] * ds_nm * delta_T  # Chain rule factor: * delta_T

    # 1st derivatives of z
    dz_dr = cp.sum(cp.conj(A) * dB_dr)
    dz_dth = cp.sum(cp.conj(A) * dB_dth)
    dz_dtau_ratio = cp.sum(cp.conj(A) * dB_dtau_ratio)

    grad_z = cp.array([[dz_dr], [dz_dth], [dz_dtau_ratio]], dtype=cp.complex128)

    # 2nd derivatives of B_nm
    d2B_dr2 = d2d_dr2[:, None] * a[:, None] * P_nm + (dd_dr[:, None]**2) * a[:, None] * Q_nm
    d2B_drdth = d2d_drdth[:, None] * a[:, None] * P_nm + (dd_dr[:, None] * dd_dth[:, None]) * a[:, None] * Q_nm
    d2B_dth2 = d2d_dth2[:, None] * a[:, None] * P_nm + (dd_dth[:, None]**2) * a[:, None] * Q_nm

    d2B_drdtau_ratio = dd_dr[:, None] * a[:, None] * R_nm * delta_T
    d2B_dthdtau_ratio = dd_dth[:, None] * a[:, None] * R_nm * delta_T
    d2B_dtau_ratio2 = a[:, None] * d2s_nm * (delta_T**2)

    # 2nd derivatives of z
    d2z_dr2 = cp.sum(cp.conj(A) * d2B_dr2)
    d2z_drdth = cp.sum(cp.conj(A) * d2B_drdth)
    d2z_dth2 = cp.sum(cp.conj(A) * d2B_dth2)
    d2z_drdtau_ratio = cp.sum(cp.conj(A) * d2B_drdtau_ratio)
    d2z_dthdtau_ratio = cp.sum(cp.conj(A) * d2B_dthdtau_ratio)
    d2z_dtau_ratio2 = cp.sum(cp.conj(A) * d2B_dtau_ratio2)

    hess_z = cp.array([
        [d2z_dr2,          d2z_drdth,        d2z_drdtau_ratio],
        [d2z_drdth,        d2z_dth2,         d2z_dthdtau_ratio],
        [d2z_drdtau_ratio, d2z_dthdtau_ratio, d2z_dtau_ratio2]
    ], dtype=cp.complex128)

    # Real Gradient (3x1) and Hessian (3x3)
    gradient = 2.0 * cp.real(cp.conj(z) * grad_z)
    hessian = 2.0 * cp.real(cp.conj(z) * hess_z + grad_z @ grad_z.T.conj())

    return hessian, gradient


def polar_to_xy(R, theta_deg):
    t = cp.deg2rad(theta_deg)
    return float(R * cp.sin(t)), float(R * cp.cos(t))


def get_RThetaTauNewton(rx, antenna_x, R, theta, tau):

    # This function computes the Newton's Update for Range, Angle, and ToA
    # The steps include Gradient & Hession Computation 
    # A Gaussian Pulse is assumed

    # Obtain Dimensions 
    n_elem  = rx.shape[0]  # Number of Antenna Elements 
    n_s     = rx.shape[1]  # Number of Samples 


    # Compute Adjoint Rx
    antenna_xy = cp.stack([cp.asarray(antenna_x, dtype=cp.float64),
                               cp.zeros(len(antenna_x), dtype=cp.float64)], axis = 1)
    
    MLE_hess, MLE_grad = f_RTheTau(rx, R, theta, tau, antenna_xy, SYSTEM.fc, SYSTEM.dt, SYSTEM.gaussian_sigma)

    # Solve for Newton's Method
    delta_r_theta_tau      = cp.linalg.solve(MLE_hess, MLE_grad)

    return (float(delta_r_theta_tau[0].item()),
            float(delta_r_theta_tau[1].item()),
            float(delta_r_theta_tau[2].item()),
            float(cp.linalg.norm(MLE_grad).item()))

def get_PowerFromRtheTau(rx, R_est, theta_est, tau_est, 
                         antenna_x, 
                         f_c=SYSTEM.fc, 
                         delta_T=SYSTEM.dt, 
                         sigma=SYSTEM.gaussian_sigma):

    delays          = propagation_delays(R_est, theta_est, antenna_x=antenna_x)
    a               = cp.exp(-1j * 2 * cp.pi * f_c * delays)
    
    # Time indices & argument matrix T_nm (N, M)
    _, M            = rx.shape
    m_idx           = cp.arange(-(M-1)/2, (M-1)/2 + 1)
    T_nm            = (m_idx[None, :] - tau_est) * delta_T - delays[:, None]

    # Gaussian pulse s(T) and derivatives
    s_nm = cp.exp(-0.5 * (T_nm / sigma)**2)

    # Complex matrix B
    B = a[:, None] * s_nm
    InnerProduct    = cp.abs(cp.sum(cp.conj(rx) * B))**2
    
    return InnerProduct



def get_Theta(rx, antenna_x, f_c = SYSTEM.fc):

    N           = len(antenna_x)
    n_theta     = N
    r_tgt       = (PM_R_MIN + PM_R_MAX)

    # Assembly Theta Axis 
    theta_axis  = cp.linspace(PM_THETA_MIN, PM_THETA_MAX, n_theta)
    th_rad_axis = cp.radians(theta_axis)
    sin_th      = cp.sin(th_rad_axis)
    cos_th      = cp.cos(th_rad_axis)

    # Assemble Delay Tensor
    d           = cp.sqrt((r_tgt * sin_th[None, :] - \
                            antenna_x[:, None])**2 + (r_tgt * cos_th[None, :])**2)     
    d_centered  = d - cp.mean(d, axis = 0, keepdims=True) 
    a           = cp.exp(-1j * 2 * cp.pi * SYSTEM.fc * d_centered / C)

    # Generate T Matrix:
    _, M                = rx.shape
    m_idx               = cp.arange(-(M-1)/2, (M-1)/2 + 1)
    T_nm_The           = m_idx[None, : ,None]*SYSTEM.dt - d_centered[:, None, :]/C
    
    # Gaussian pulse s(T) and Phase
    s_nm = cp.exp(-0.5 * (T_nm_The / SYSTEM.gaussian_sigma)**2)
    B    = s_nm * a[:, None, :]

    # Assemble Profile
    theta_profile       = cp.abs(cp.einsum('nm, nmk -> k', rx, B.conj()))**2

    # Get 3-point parabola fitting for theta 
    max_idx            = theta_profile.argmax()
    theta_fitted, _    = find_parabola_vertex(theta_axis[max_idx-1:max_idx+2], 
                                              theta_profile[max_idx-1:max_idx+2])

    return theta_fitted

def get_RFromTheta(rx, antenna_x, theta_guess):

    th_rad              = cp.radians(theta_guess)
    sin_th, cos_th      = cp.sin(th_rad), cp.cos(th_rad)
    r_axis              = PM_R_GRID
    d                   = cp.sqrt((r_axis[None, :] * sin_th - antenna_x[:, None])**2 + (r_axis[None, :] * cos_th)**2) 
    d_centered          = d - cp.mean(d, axis = 0, keepdims=True) 
    a                   = cp.exp(-1j * 2 * cp.pi * SYSTEM.fc * d_centered / C)

    # Centering Distances:
    _, M                = rx.shape
    m_idx               = cp.arange(-(M-1)/2, (M-1)/2 + 1)
    T_nm_R              = m_idx[None, :, None]*SYSTEM.dt - d_centered[:, None, :]/C

    # Gaussian pulse s(T) and derivatives
    s_nm = cp.exp(-0.5 * (T_nm_R / SYSTEM.gaussian_sigma)**2)

    r_profile           = cp.abs(cp.einsum('nm, nml -> l', rx, (s_nm*a[:, None, :]).conj()))


    return r_axis[r_profile.argmax()]

def get_TauFromRT(rx, r, theta, antenna_x, dt): 
    # Geometry & steering vector derivatives
    th_rad              = cp.radians(theta)
    sin_th, cos_th      = cp.sin(th_rad), cp.cos(th_rad)
    d                   = cp.sqrt((r * sin_th - antenna_x)**2 + (r * cos_th)**2) 
    a                   = cp.exp(-1j * 2 * cp.pi * SYSTEM.fc * d / C)
    
    # Time indices & argument matrix T_nm (N, M)
    tau_range       = 10
    tau_axis        = cp.linspace(-tau_range, tau_range, tau_range*4)
    _, M            = rx.shape
    m_idx           = cp.arange(-(M-1)/2, (M-1)/2 + 1)
    tau_axis        = -cp.mean(d / (C * dt)) + tau_axis
    T_nm            = (m_idx[None, :, None] - tau_axis[None, None, :]) * dt - d[:, None, None]/C

    # Gaussian pulse s(T) and derivatives
    s_nm = cp.exp(-0.5 * (T_nm / SYSTEM.gaussian_sigma)**2)

    # Complex matrix B
    B = a[:, None, None] * s_nm
    tau_profile    = cp.abs(cp.einsum('ij, ijk -> k', rx, B.conj()))**2
    
    return tau_axis[tau_profile.argmax()]

# ============================================================
# Complete pipeline
# ============================================================

def localize_MLE(rx, time_axis, antenna_x, theta_hint=None, tau_hint=None, R_hint = None, 
                           use_radon = False):

    antenna_x_gpu    = cp.asarray(antenna_x)
    rx_gpu           = cp.asarray(rx)
    if use_radon:
        # Hinted Values Are Unused
        # Use Radon Transform To Obtain Initial Guess
        results       = localize(rx, time_axis, antenna_x, theta_hint, tau_hint)
        R_init        = results.R 
        theta_init    = results.theta
        tau_init      = get_TauFromRT(rx_gpu, R_init, theta_init, antenna_x_gpu, SYSTEM.dt)

    else:
        theta_init      = get_Theta(rx_gpu, antenna_x_gpu) if theta_hint is None else theta_hint
        R_init          = get_RFromTheta(rx_gpu, antenna_x_gpu, theta_init) if R_hint is None else R_hint 
        tau_init        = get_TauFromRT(rx_gpu, R_init, theta_init, antenna_x_gpu, SYSTEM.dt)
    
    if PRNT:
        print(f"Initial Range Estimated at: {R_init:.2f} !!")

    # Define Iteration Parameters 
    # =======================================================
    max_iter    = 50                        # maximum iterations 
    grad_tol    = 1e-11                     # tolerance of gradient
    step_size   = 0.8                       # Step-size of Newton's Update 
    R_est       = R_init                    # Define initial R
    power_delta_tol = -1e-6                 # Tolerance of Power Delta
    theta_est   = theta_init                # Define initial Theta
    power_hist  = 0                         # Used to Compute Power Delta
    tau_est     = tau_init                  # Tau Estimate                          

    for it in range(max_iter): 
        
        R_delta, theta_delta, tau_delta, grad_norm = get_RThetaTauNewton(rx_gpu, 
                                            antenna_x_gpu, 
                                            R_est, 
                                            theta_est, 
                                            tau_est) 

        if grad_norm <= grad_tol:
            if PRNT:
                print(f"Iteration Stop with Grad: {grad_norm:.3e} !!")
            break
        else: 
            R_est_new           = R_est - step_size * R_delta 
            theta_est_new       = theta_est - step_size * theta_delta 
            tau_est_new         = tau_est - step_size * tau_delta

        
        power_new = get_PowerFromRtheTau(rx_gpu, 
                                         R_est_new, 
                                         theta_est_new, 
                                         tau_est_new,
                                         antenna_x_gpu)

        # breakpoint()
        power_delta = (power_new-power_hist)/(power_new + power_hist)

        if PRNT:
            print(f"Iter. {it:3d}, "
                f"Power Delta {power_delta:.3e}, "
                f"R: {R_est_new: .2f},  "
                f"The: {theta_est_new: .2f},  "
                f"Tau: {tau_est_new: .2f},  "
                f"rDelta: {abs(R_delta): .2e},  "
                f"ThDelta: {abs(theta_delta): .2e},  "
                f"TaDelta: {abs(tau_delta): .2e},  "
                f"Grad_Norm: {grad_norm: .2e} !!")

        if power_delta < power_delta_tol: 
            tau_est      = get_TauFromRT(rx_gpu, R_est, theta_est, antenna_x_gpu, SYSTEM.dt)
            continue

        power_hist = power_new
        R_est      = R_est_new
        theta_est  = theta_est_new
        tau_est    = tau_est_new

    # cp.exp(-1j * 2.0 * cp.pi * SYSTEM.fc * delay_axis_centered)
    tau_hat         = tau_est * SYSTEM.dt
    x_hat, y_hat    = polar_to_xy(R_est, theta_est)   

    return LocalizationResult(R_est, theta_est, tau_hat, x_hat, y_hat,
                              0,0,0,0,0,0)
